import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from benchmark.long_bench import pred as longbench_pred
from benchmark.long_bench.metrics import classification_score, qa_f1_score
from benchmark.sparsevllm_regression.manifest import load_manifest, resolve_method_config


class LongBenchDeltaKVContractsTest(unittest.TestCase):
    def _omnikv_args(self, hyper_param):
        return SimpleNamespace(
            hyper_param=json.dumps(hyper_param),
            sparse_method="omnikv",
            max_model_len=32768,
            deltakv_checkpoint_path=None,
            allow_single_omnikv_full_layer=False,
        )

    def test_longbench_requires_explicit_prefix_caching_opt_in(self):
        args = self._omnikv_args(
            resolve_method_config(
                load_manifest()["methods"]["omnikv"],
                model_id="qwen25_7b",
                require_model_config=True,
            )
        )
        config = longbench_pred._build_infer_config(args)
        self.assertIs(config["enable_prefix_caching"], False)

        requested = json.loads(args.hyper_param)
        requested["enable_prefix_caching"] = True
        args.hyper_param = json.dumps(requested)
        with self.assertRaisesRegex(ValueError, "enable_prefix_caching=False"):
            longbench_pred._build_infer_config(args)

        args.allow_prefix_caching = True
        config = longbench_pred._build_infer_config(args)
        self.assertIs(config["enable_prefix_caching"], True)

    def test_longbench_records_requested_and_effective_omnikv_config(self):
        config = resolve_method_config(
            load_manifest()["methods"]["omnikv"],
            model_id="qwen25_7b",
            require_model_config=True,
        )
        args = self._omnikv_args(config)
        infer_config = longbench_pred._build_infer_config(args)
        requested = longbench_pred._requested_runtime_config(args, infer_config)
        requested_layers = requested["config"]["full_attention_layers"]
        self.assertTrue(requested_layers)

        runtime_info = {
            "sparse_method": "omnikv",
            "full_attention_layers": requested_layers,
        }
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                worker_info=lambda **_kwargs: runtime_info,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            resolved = Path(tmp) / "resolved_config.json"
            resolved.write_text(
                json.dumps({"backend": "sparsevllm", "requested": requested}),
                encoding="utf-8",
            )
            longbench_pred._record_effective_runtime_config(
                generate_fn=generate_fn,
                out_root=tmp,
            )
            recorded = json.loads(resolved.read_text(encoding="utf-8"))

        self.assertEqual(recorded["requested"], requested)
        self.assertEqual(recorded["effective_runtime"], runtime_info)
        self.assertEqual(
            recorded["requested"]["config"]["full_attention_layers"],
            recorded["effective_runtime"]["full_attention_layers"],
        )

    def test_longbench_records_final_prefix_cache_statistics(self):
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                worker_load=lambda: {
                    "active_requests": 0,
                    "cache": {
                        "prefix_cache_hit_requests": 3,
                        "prefix_cache_hit_tokens": 48,
                    },
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            longbench_pred._write_worker_load_stats(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=0,
            )
            recorded = json.loads(
                (Path(tmp) / "worker_load_stats_rank0.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            recorded["worker_load"]["cache"]["prefix_cache_hit_tokens"],
            48,
        )

    def test_chat_template_policy_matches_regular_prompt_paths(self):
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="off")
        )
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="on_strip")
        )
        self.assertFalse(
            longbench_pred.should_use_chat_template("hotpotqa", no_chat_template=True)
        )

    def test_hotpotqa_and_trec_metric_contracts(self):
        self.assertEqual(qa_f1_score("Paris", "Paris"), 1.0)
        self.assertEqual(qa_f1_score("Paris", "London"), 0)
        self.assertEqual(
            classification_score(
                "DESC",
                "DESC",
                all_classes=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
            ),
            1.0,
        )

    def test_longbench_data_validation_fails_fast_for_missing_hotpotqa_and_trec(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = longbench_pred.DATA_PREFIX_PATH
            longbench_pred.DATA_PREFIX_PATH = str(Path(tmp) / "missing")
            try:
                with self.assertRaisesRegex(FileNotFoundError, "LongBench data root"):
                    longbench_pred.validate_longbench_data_paths(["hotpotqa", "trec"], use_longbench_e=False)
            finally:
                longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_data_validation_requires_explicit_root(self):
        old_root = longbench_pred.DATA_PREFIX_PATH
        longbench_pred.DATA_PREFIX_PATH = None
        try:
            with self.assertRaisesRegex(FileNotFoundError, "SPARSEVLLM_LONGBENCH_DATA_DIR"):
                longbench_pred.validate_longbench_data_paths(["hotpotqa"], use_longbench_e=False)
        finally:
            longbench_pred.DATA_PREFIX_PATH = old_root

    def test_sparsevllm_data_workers_receive_distinct_master_ports(self):
        launched = []

        class Process:
            def wait(self):
                return 0

        def fake_popen(command, *, env, cwd):
            launched.append((command, env, cwd))
            return Process()

        worker_args = SimpleNamespace(ws=4)
        with (
            patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "SPARSEVLLM_MASTER_PORT": "24300",
                },
                clear=False,
            ),
            patch.object(longbench_pred.subprocess, "Popen", side_effect=fake_popen),
        ):
            longbench_pred.launch_single_gpu_workers(worker_args, "/tmp/longbench-output")

        self.assertEqual(
            [env["CUDA_VISIBLE_DEVICES"] for _command, env, _cwd in launched],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            [env["SPARSEVLLM_MASTER_PORT"] for _command, env, _cwd in launched],
            ["24300", "24301", "24302", "24303"],
        )

    def test_longbench_records_actual_decode_cuda_graph_state(self):
        graph_runner = SimpleNamespace(
            _graphs={
                "captured": SimpleNamespace(graph=object()),
                "uncaptured": SimpleNamespace(graph=None),
            },
            last_state_key="captured",
            capture_count=2,
            replay_count=7,
            eager_static_count=0,
            force_eager_count=0,
        )
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                config=SimpleNamespace(decode_graph=True),
                model_runner=SimpleNamespace(
                    decode_graph_runner=graph_runner,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            status = longbench_pred._write_decode_cuda_graph_status(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=2,
            )
            path = Path(tmp) / "decode_graph_status_rank2.json"

            self.assertEqual(status["rank"], 2)
            self.assertTrue(status["configured"])
            self.assertTrue(status["runner_initialized"])
            self.assertEqual(status["state_count"], 2)
            self.assertEqual(status["graph_count"], 1)
            self.assertTrue(status["active"])
            self.assertEqual(status["capture_count"], 2)
            self.assertEqual(status["replay_count"], 7)
            self.assertEqual(status["last_state_key"], "captured")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                status,
            )

    def test_longbench_records_business_graph_counter_delta(self):
        graph_runner = SimpleNamespace(
            _graphs={"captured": SimpleNamespace(graph=object())},
            last_state_key="captured",
            capture_count=1,
            replay_count=3,
            eager_static_count=0,
            force_eager_count=0,
        )
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                config=SimpleNamespace(decode_graph=True),
                model_runner=SimpleNamespace(
                    decode_graph_runner=graph_runner,
                ),
            )
        )
        before = longbench_pred._decode_cuda_graph_status(
            generate_fn=generate_fn,
            rank=0,
        )
        graph_runner.replay_count = 11

        with tempfile.TemporaryDirectory() as tmp:
            status = longbench_pred._write_decode_cuda_graph_status(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=0,
                before=before,
            )

        self.assertEqual(status["before"]["replay_count"], 3)
        self.assertEqual(status["replay_count"], 11)
        self.assertEqual(status["counter_delta"]["replay_count"], 8)
        self.assertEqual(status["counter_delta"]["eager_static_count"], 0)
        self.assertEqual(status["counter_delta"]["force_eager_count"], 0)

    def test_longbench_fails_if_sparsevllm_graph_state_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "_sparsevllm_llm"):
                longbench_pred._write_decode_cuda_graph_status(
                    generate_fn=object(),
                    out_root=tmp,
                    rank=0,
                )

def test_resume_rebuilds_outputs_from_committed_samples_and_retries_failures(tmp_path):
    record = dict(dataset='trec', source_idx=0, sample_idx=0, status='success',
                  pred='label', raw_pred='label', answers=['label'], all_classes=['label'], prompt_tokens=1)
    longbench_pred._write_sample_record(out_root=str(tmp_path), task_out_path=str(tmp_path/'trec.jsonl'), record=record)
    failed = dict(record, source_idx=1, sample_idx=1, status='model_failed', error='interrupted')
    longbench_pred._write_sample_record(out_root=str(tmp_path), task_out_path=str(tmp_path/'trec.jsonl'), record=failed)
    # Process termination may leave one torn commit and inconsistent derived files.
    with (tmp_path/'sample_results.jsonl').open('ab') as f:f.write(b'{"dataset":')
    (tmp_path/'trec.jsonl').write_text('partial derived file')
    assert longbench_pred._resume_samples(str(tmp_path), ['trec']) == {('trec', 0)}
    assert longbench_pred._read_worker_jsonl(tmp_path/'sample_results.jsonl') == [record]
    assert longbench_pred._read_worker_jsonl(tmp_path/'trec.jsonl')[0]['pred'] == 'label'
    assert json.loads((tmp_path/'resume_attempts.jsonl').read_text())['retry_samples'] == [failed]
    assert longbench_pred._resume_samples(str(tmp_path), ['trec']) == {('trec', 0)}


@pytest.mark.parametrize('contents', [b'{broken}\n{}\n', b'{"dataset":"trec","source_idx":0,"status":"unknown"}\n'])
def test_resume_rejects_corrupt_committed_records_without_overwriting(tmp_path, contents):
    path=tmp_path/'sample_results.jsonl';path.write_bytes(contents)
    with pytest.raises(ValueError):longbench_pred._resume_samples(str(tmp_path), ['trec'])
    assert path.read_bytes() == contents


def test_resume_contract_rejects_changed_weights_data_code_or_settings(tmp_path, monkeypatch):
    code=tmp_path/'src/sparsevllm/method.py';code.parent.mkdir(parents=True);code.write_text('original')
    data=tmp_path/'data/trec.jsonl';data.parent.mkdir();data.write_text('{}\n')
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'weights')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    monkeypatch.setattr(longbench_pred,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(longbench_pred,'DATA_PREFIX_PATH',str(tmp_path))
    args=SimpleNamespace(e=False,model_path=str(model),tokenizer_path=None,deltakv_checkpoint_path=None,
                         resume=False,output_root=str(tmp_path),num_samples=20,seed=42)
    config=dict(leasesparse_predictor_path=str(checkpoint))
    contract=longbench_pred._resume_contract(args,['trec'],config)
    longbench_pred._prepare_resume_contract(tmp_path,contract,False)
    original=(tmp_path/'resume_contract.json').read_bytes()
    args.resume=True
    longbench_pred._prepare_resume_contract(tmp_path,longbench_pred._resume_contract(args,['trec'],config),True)
    for path in (data,checkpoint,code):
        saved=path.read_bytes();path.write_bytes(saved+b'changed')
        with pytest.raises(ValueError,match='changed'):
            longbench_pred._prepare_resume_contract(tmp_path,longbench_pred._resume_contract(args,['trec'],config),True)
        assert (tmp_path/'resume_contract.json').read_bytes()==original
        path.write_bytes(saved)
    args.seed=43
    with pytest.raises(ValueError,match='changed'):
        longbench_pred._prepare_resume_contract(tmp_path,longbench_pred._resume_contract(args,['trec'],config),True)


@pytest.mark.parametrize('world_size',[1,2])
def test_interrupted_longbench_resumes_only_missing_samples_and_scores_once(tmp_path,monkeypatch,world_size):
    data=tmp_path/'data';data.mkdir()
    rows=[dict(context=str(i),answers=[str(i)],all_classes=[],length=1) for i in range(5)]
    (data/'trec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(longbench_pred,'DATA_PREFIX_PATH',str(tmp_path))
    tokenizer=SimpleNamespace(bos_token=None,encode=lambda text,**kw:[int(text)])
    calls=[];stop=[True]
    def generate(prompts,**kw):
        if stop[0] and len(calls)==1:raise KeyboardInterrupt()
        calls.extend(p[0] for p in prompts)
        return [str(p[0]) for p in prompts]
    loads=[]
    def load(*a):loads.append(1);return generate,tokenizer,100,[]
    monkeypatch.setattr(longbench_pred,'load_model_and_tokenizer',load)
    for name in ('_record_effective_runtime_config','_write_decode_cuda_graph_status',
                 '_write_operator_runtime_stats','_write_worker_load_stats'):
        monkeypatch.setattr(longbench_pred,name,lambda **kw:None)
    monkeypatch.setattr(longbench_pred,'_decode_cuda_graph_status',lambda **kw:{})
    args=SimpleNamespace(resume=False,seed=42,e=False,num_samples=5,min_prompt_tokens=None,
                         batch_size=1,max_new_tokens_override=None,no_chat_template=True,
                         thinking_mode='off',temperature=0,top_p=1,top_k=1)
    out=tmp_path/'resumed';out.mkdir()
    def run(rank,root):
        longbench_pred.worker(rank,world_size,['trec'],{'trec':'{context}'},{'trec':4},args,str(root),100,{})
    with pytest.raises(KeyboardInterrupt):run(0,out)
    assert calls==[0]
    args.resume=True;stop[0]=False
    for rank in range(world_size):run(rank,out)
    assert sorted(calls)==list(range(5))
    if world_size>1:longbench_pred._merge_worker_outputs(str(out),datasets=['trec'],world_size=world_size)
    resumed=longbench_pred._read_worker_jsonl(out/'sample_results.jsonl')
    assert len(resumed)==5 and all(r['status']=='success' for r in resumed)
    load_count=len(loads)
    for rank in range(world_size):run(rank,out)
    assert len(loads)==load_count  # Fully complete work never loads the model again.
    args.resume=False
    uninterrupted=tmp_path/'uninterrupted';uninterrupted.mkdir()
    for rank in range(world_size):run(rank,uninterrupted)
    if world_size>1:longbench_pred._merge_worker_outputs(str(uninterrupted),datasets=['trec'],world_size=world_size)
    assert resumed==longbench_pred._read_worker_jsonl(uninterrupted/'sample_results.jsonl')


def test_longbench_wrapper_keeps_interrupted_progress_and_cleans_only_after_archival(tmp_path,monkeypatch):
    import importlib.util
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts/tmp/run_lease_longbench20.py'
    spec=importlib.util.spec_from_file_location('lease_longbench_wrapper',script)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    tasks=json.loads((script.parents[2]/'benchmark/long_bench/config/dataset2maxlen.json').read_text())
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{"num_hidden_layers":28}')
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'checkpoint')
    data=tmp_path/'data';data.mkdir()
    for task in tasks:(data/f'{task}.jsonl').write_text('{}\n'*20)
    work=tmp_path/'work';work.mkdir();(work/'unrelated.txt').write_text('preserve')
    report=tmp_path/'record.md'
    monkeypatch.setattr(sys,'argv',[str(script),'--model',str(model),'--checkpoint',str(checkpoint),
        '--data-root',str(tmp_path),'--report',str(report),'--work-dir',str(work)])
    monkeypatch.setattr(wrapper.subprocess,'check_output',lambda *a,**kw:'')
    commands=[]
    def launch(cmd,**kwargs):
        commands.append(cmd)
        out=Path(cmd[cmd.index('--output_root')+1])
        if len(commands)==1:
            (out/'resume_contract.json').write_text('{}')
            (out/'sample_results.jsonl').write_text('committed progress')
            raise subprocess.CalledProcessError(1,cmd)
        if out.name=='predictor4':
            assert '--resume' in cmd
            assert (out/'sample_results.jsonl').read_text()=='committed progress'
        records=[dict(dataset=t,source_idx=i,prompt_tokens=10) for t in tasks for i in range(20)]
        (out/'sample_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'resolved_config.json').write_text('{}')
        score=dict(status='success',task_statuses={t:dict(status_counts={'success':20}) for t in tasks})
        score.update({t:1.0 for t in tasks})
        (out/'result.json').write_text(json.dumps(score))
    monkeypatch.setattr(wrapper.subprocess,'run',launch)
    with pytest.raises(subprocess.CalledProcessError):wrapper.main()
    assert (work/'predictor4/sample_results.jsonl').read_text()=='committed progress'
    wrapper.main()
    assert len(commands)==3
    assert not (work/'predictor4').exists() and not (work/'ema1').exists()
    assert (work/'unrelated.txt').read_text()=='preserve'
    assert '两方法对照' in report.read_text()


if __name__ == "__main__":
    unittest.main()
