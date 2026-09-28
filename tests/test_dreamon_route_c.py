"""Route C reference isolation, gradients, edits and checkpoint tests (CPU)."""
import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from cosyvoice.llm.dreamon_route_c import DynamicSpeechLM, RouteCTrainer, FILL, EXPAND, DELETE
from cosyvoice.llm.dreamon_route_b import DynamicSpeechLM as BModel, EditGenerationError
from cosyvoice.dataset.dreamon_reference import pair_cache, validate_reference_cache
from cosyvoice.bin.train_route_c import batch
from test_dreamon_lora import AttentionBackbone
from test_dreamon_speech import TinyTokenizer


def bridge(cls=DynamicSpeechLM):
    torch.manual_seed(123)
    return cls(backbone=AttentionBackbone(),tokenizer=TinyTokenizer(),
        speech_embedding=nn.Embedding(10,2),speech_decoder=nn.Linear(2,10),task_embedding=nn.Embedding(2,2),
        mask_token_id=7,bos_token_id=8,eos_token_id=9,speech_vocab_size=7,seed=123,max_sequence_tokens=128)


def record():
    return dict(utt="1_2_3_4",text="abc",text_ids=[2,3,4],speech=[0,1,2,3],
                ref_utt="1_2_3_5",ref_text="de",ref_text_ids=[0,1],ref_speech=[4,5,6])


class RouteCTests(unittest.TestCase):
    def test_reference_reaches_backbone_in_expected_order(self):
        m=bridge(); r=record(); ids=m.encode_text(r["text"],r["ref_text"])
        ref=torch.tensor([r["ref_speech"]]);canvas=torch.tensor([[-1,2]])
        with patch.object(m.backbone,"forward",wraps=m.backbone.forward) as call:
            s,a=m.edit_forward(ids,canvas,ref)
        x=call.call_args.kwargs["inputs_embeds"]
        self.assertEqual(x.shape[1],1+5+1+3+2+1)
        torch.testing.assert_close(x[:,1:6],m.backbone.get_input_embeddings()(ids))
        torch.testing.assert_close(x[:,7:10],m.speech_in_proj(m.speech_embedding(ref)))
        self.assertEqual(s.shape,(1,2,7));self.assertEqual(a.shape,(1,2,3))
        s2,a2=m.edit_forward(ids,canvas,torch.tensor([[0,0,0]]))
        self.assertFalse(torch.allclose(s,s2));self.assertFalse(torch.allclose(a,a2))

    def test_training_masks_and_loss_only_target_and_has_gradients(self):
        m=bridge(); t=RouteCTrainer(m,use_lora=True,lora_rank=2,lora_alpha=4,lora_dropout=0.)
        t.eval();t.validation_edits=False;t.validation_mask_ratio=1.
        r=record()
        with patch.object(m,"edit_forward",wraps=m.edit_forward) as call:
            out=t(batch(r),"cpu")
        ids,canvas,prompt=call.call_args.args
        self.assertEqual(ids.tolist(),[r["ref_text_ids"]+r["text_ids"]])
        self.assertEqual(prompt.tolist(),[r["ref_speech"]])
        self.assertTrue(canvas.eq(-1).all())
        self.assertEqual(float(out["token_count"]),len(r["speech"]))
        out["loss"].backward()
        for name,p in t.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertEqual({id(p) for p in t.parameters() if p.requires_grad},
                         {id(p) for g in t.optimizer_parameter_groups() for p in g["params"]})

    def test_edits_never_change_reference(self):
        m=bridge();ref=torch.tensor([[4,5,6]]);original=ref.clone();actions=iter([EXPAND,FILL,DELETE,FILL])
        def forward(ids,canvas,prompt):
            torch.testing.assert_close(prompt,original)
            self.assertEqual(ids.tolist(),m.encode_text("abc","de").tolist())
            a=torch.full((1,canvas.numel(),3),-20.);a[:,:,next(actions)]=20.
            s=torch.full((1,canvas.numel(),7),-20.);s[:,:,0]=20.
            return s,a
        with patch.object(m,"edit_forward",side_effect=forward):
            tokens,report=m.generate_dynamic("abc",prompt_text="de",prompt_speech_tokens=ref,
                initial_masks=2,maximum=8,max_steps=4,temperature=0,action_temperature=0)
        self.assertEqual(tokens.tolist(),[[0,0]])
        torch.testing.assert_close(ref,original)
        self.assertEqual(report["reference_speech_tokens"],3)
        self.assertFalse(report["target_audio_used"])
        self.assertFalse(report["duration_predictor_used"])

    def test_reference_required_and_context_budget(self):
        m=bridge()
        with self.assertRaises(ValueError):
            m.generate_dynamic("a",prompt_text="",prompt_speech_tokens=torch.tensor([[1]]))
        with self.assertRaises(ValueError):
            m.generate_dynamic("a",prompt_text="b",prompt_speech_tokens=torch.empty(1,0,dtype=torch.long))
        m.max_sequence_tokens=8
        with self.assertRaisesRegex(ValueError,"context"):
            m.generate_dynamic("a",prompt_text="b",prompt_speech_tokens=torch.tensor([[1,2]]),initial_masks=2)
        b=batch(record());b["ref_utts"]=b["utts"]
        with self.assertRaises(ValueError):RouteCTrainer(bridge())(b,"cpu")

    def test_checkpoint_roundtrip_and_b_c_cannot_be_confused(self):
        m=bridge();t=RouteCTrainer(m,use_lora=True,lora_rank=2,lora_alpha=4,lora_dropout=0.)
        t(batch(record()),"cpu")["loss"].backward()
        payload=t.training_checkpoint(dict(epoch=0,step=1))
        self.assertEqual(payload["format_version"],7)
        self.assertNotIn("route_b",payload)
        restored=bridge();restored.load_adapter_checkpoint(payload)
        m.eval();restored.eval()
        args=(m.encode_text("abc","de"),torch.tensor([[-1,2]]),torch.tensor([[4,5,6]]))
        for x,y in zip(m.edit_forward(*args),restored.edit_forward(*args)):torch.testing.assert_close(x,y)
        with self.assertRaises(ValueError):bridge(BModel).load_adapter_checkpoint(payload)
        with self.assertRaises(ValueError):restored.load_adapter_checkpoint(bridge(BModel).initial_adapter_checkpoint())

    def test_pairing_split_isolation_and_reproducibility(self):
        def rows(spk):
            return [dict(utt=f"{spk}_1_1_{i}",text=f"text {i}",text_ids=[i+1],speech=[i]*3) for i in range(3)]
        src=dict(train=rows(1),dev=rows(2),summary={})
        a=pair_cache(src,128);b=pair_cache(src,128)
        self.assertEqual(a,b)
        for split in ("train","dev"):
            for r in a[split]:self.assertNotEqual(r["utt"],r["ref_utt"])
        bad=copy.deepcopy(a);bad["train"][0]["ref_speech"]=[6]
        with self.assertRaises(ValueError):validate_reference_cache(bad,128)
        src["dev"]=src["train"]
        with self.assertRaises(ValueError):pair_cache(src,128)


if __name__=="__main__":unittest.main()
