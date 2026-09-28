"""Route B corruption, true edit trajectories, loss and checkpoint tests."""
import copy
import random
import unittest
from unittest.mock import patch
import torch
from torch import nn
from cosyvoice.llm.dreamon_route_b import (DynamicSpeechLM,RouteBTrainer,corrupt_speech,
                                         EditGenerationError,FILL,EXPAND,DELETE)
from test_dreamon_speech import TinyTokenizer
from test_dreamon_lora import AttentionBackbone
from test_dreamon_training import training_batch

def bridge():
    torch.manual_seed(123)
    return DynamicSpeechLM(backbone=AttentionBackbone(),tokenizer=TinyTokenizer(),
        speech_embedding=nn.Embedding(10,2),speech_decoder=nn.Linear(2,10),task_embedding=nn.Embedding(2,2),
        mask_token_id=7,bos_token_id=8,eos_token_id=9,speech_vocab_size=7,seed=123,max_sequence_tokens=128)

def scripted(actions):
    calls=iter(actions)
    def forward(ids,canvas):
        a=torch.full((1,canvas.numel(),3),-20.)
        a[:,:,next(calls)]=20.
        s=torch.full((1,canvas.numel(),7),-20.);s[:,:,0]=20.
        return s,a
    return forward

class RouteBTests(unittest.TestCase):
    def test_corruption_preserves_all_true_tokens_including_repeats(self):
        speech=torch.tensor([[0,0,1,2,2,2,3,4]*8])
        seen=set()
        for seed in range(100):
            canvas,actions,labels,info=corrupt_speech(speech,.8,random.Random(seed),100)
            self.assertEqual(sum(info["latent_span_lengths"]),speech.numel())
            self.assertTrue(torch.equal(canvas.eq(-1),actions.ne(-100)))
            self.assertTrue(torch.equal(labels.ne(-100),actions.eq(FILL)))
            seen.update(actions.flatten().tolist())
        self.assertTrue({FILL,EXPAND,DELETE}.issubset(seen))

    def test_uncompressed_target_labels_reconstruct_repeated_speech(self):
        speech=torch.tensor([[0,0,2,2,0,3]])
        c,a,l,info=corrupt_speech(speech,.5,random.Random(3),20,merge_rounds=0)
        restored=c.clone();restored[a.eq(FILL)]=l[a.eq(FILL)]
        torch.testing.assert_close(restored[a.ne(DELETE)].reshape(1,-1),speech)

    def test_grows_shrinks_and_resolves_valid_token_zero(self):
        model=bridge()
        with patch.object(model,"edit_forward",side_effect=scripted([EXPAND,FILL,DELETE,FILL])):
            tokens,r=model.generate_dynamic("a",initial_masks=2,maximum=8,temperature=0,action_temperature=0,max_steps=4)
        self.assertEqual(tokens.tolist(),[[0,0]])
        self.assertEqual([x["length_after"] for x in r["steps"]],[3,3,2,2])
        self.assertEqual(r["expand_count"],1);self.assertEqual(r["delete_count"],1)
        self.assertFalse(r["duration_predictor_used"])

    def test_caps_empty_and_budget_are_failures_not_silent_audio(self):
        for acts,kwargs,reason in [
            ([EXPAND],dict(initial_masks=2,maximum=2),"canvas_cap_reached"),
            ([DELETE],dict(initial_masks=1,maximum=3),"empty_output"),
            ([EXPAND],dict(initial_masks=1,maximum=3,max_steps=1),"edit_step_budget_exhausted")]:
            m=bridge()
            with patch.object(m,"edit_forward",side_effect=scripted(acts)):
                with self.assertRaises(EditGenerationError) as ctx:m.generate_dynamic("a",**kwargs)
            self.assertEqual(ctx.exception.report["failure_reason"],reason)

    def test_all_edit_losses_keep_all_trainable_branches_in_graph(self):
        m=bridge();trainer=RouteBTrainer(m,use_lora=True,lora_rank=2,lora_alpha=4,
                                       lora_dropout=0.,prompt_probability=0.)
        batch=training_batch()
        def all_edit(tokens,ratio,rng,maximum):
            c=torch.full((1,2),-1,dtype=torch.long)
            return c,torch.tensor([[EXPAND,DELETE]]),torch.full_like(c,-100),dict(original_mask_fraction=1.)
        with patch("cosyvoice.llm.dreamon_route_b.corrupt_speech",side_effect=all_edit):
            losses=trainer(batch,"cpu");losses["loss"].backward()
        for name,p in trainer.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
        expected={id(p) for p in trainer.parameters() if p.requires_grad}
        self.assertEqual(expected,{id(p) for g in trainer.optimizer_parameter_groups() for p in g["params"]})

    def test_checkpoint_roundtrip_and_bad_head_rejects_before_mutation(self):
        m=bridge();trainer=RouteBTrainer(m,use_lora=True,lora_rank=2,lora_alpha=4,lora_dropout=0.)
        with torch.no_grad():
            for p in m.edit_head.parameters():p.add_(.2)
        payload=trainer.training_checkpoint(dict(epoch=0,step=5))
        self.assertEqual(payload["format_version"],6)
        self.assertNotIn("route_a",payload)
        restored=bridge();restored.load_adapter_checkpoint(payload)
        m.eval();restored.eval()
        ids=m.encode_text("abc");canvas=torch.tensor([[-1,2,-1]])
        for a,b in zip(m.edit_forward(ids,canvas),restored.edit_forward(ids,canvas)):
            torch.testing.assert_close(a,b)
        bad=copy.deepcopy(payload)
        bad["route_b"]["edit_state_dict"]["1.weight"][0,0]=float("nan")
        before={k:v.clone() for k,v in restored.state_dict().items()}
        with self.assertRaises(ValueError):restored.load_adapter_checkpoint(bad)
        for k,v in restored.state_dict().items():torch.testing.assert_close(v,before[k])

    def test_validation_is_repeatable(self):
        t=RouteBTrainer(bridge(),prompt_probability=0.)
        t.eval();a=t(training_batch(),"cpu");b=t(training_batch(),"cpu")
        for k in a:torch.testing.assert_close(a[k],b[k])


    def test_greedy_edit_cycle_is_reported_without_burning_budget(self):
        m=bridge()
        with patch.object(m,"edit_forward",side_effect=scripted([EXPAND,DELETE])):
            with self.assertRaises(EditGenerationError) as ctx:
                m.generate_dynamic("a",initial_masks=2,maximum=8,max_steps=100,action_temperature=0.)
        self.assertEqual(ctx.exception.report["failure_reason"],"deterministic_edit_cycle")
        self.assertEqual(len(ctx.exception.report["steps"]),2)

    def test_stochastic_actions_can_revisit_then_finish(self):
        m=bridge()
        with patch.object(m,"edit_forward",side_effect=scripted([EXPAND,DELETE,FILL,FILL])):
            tokens,r=m.generate_dynamic("a",initial_masks=2,maximum=8,temperature=0.,
                                        action_temperature=.8,max_steps=4)
        self.assertEqual(tokens.tolist(),[[0,0]])
        self.assertEqual(r["speech_tokens"],2+r["expand_count"]-r["delete_count"])

    def test_noedit_validation_uses_all_true_targets(self):
        t=RouteBTrainer(bridge(),prompt_probability=0.,validation_mask_ratio=1.)
        t.eval()
        t.validation_edits=False
        b=training_batch()
        out=t(b,"cpu")
        self.assertEqual(float(out["token_count"]),float(b["speech_token_len"].float().mean()))
        self.assertEqual(float(out["expand_count"]),0.)
        self.assertEqual(float(out["delete_count"]),0.)
        self.assertAlmostEqual(float(out["canvas_ratio"]),1.)
        self.assertGreater(float(out["token_ce"]),0.)

    def test_weighted_metrics_exclude_empty_token_examples(self):
        from cosyvoice.bin.train_route_b import summarize, METRICS
        totals={k:0. for k in METRICS}
        totals.update(token_ce_sum=18.,token_count=3.,token_correct=1.,
                      action_ce_sum=20.,masked_count=10.,loss=9.)
        report=summarize([totals[k] for k in METRICS],4)
        self.assertEqual(report["token_ce"],6.)
        self.assertEqual(report["action_ce"],2.)
        self.assertAlmostEqual(report["acc"],1/3)
        self.assertEqual(report["loss"],2.25)

    def test_plain_corruption_has_no_length_supervision(self):
        target=torch.tensor([[0,0,1,2,2]])
        canvas,actions,labels,info=corrupt_speech(
            target,1.,random.Random(42),20,merge_rounds=0,surplus_count=0)
        self.assertTrue(actions.eq(FILL).all())
        torch.testing.assert_close(labels,target)
        self.assertEqual(canvas.numel(),target.numel())

if __name__=="__main__":unittest.main()
