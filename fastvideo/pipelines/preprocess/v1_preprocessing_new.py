from fastvideo.distributed import (maybe_init_distributed_environment_and_model_parallel)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


def main(fastvideo_args: FastVideoArgs) -> None:
    # Honor sp_size/tp_size from CLI so multi-GPU VAE (--vae-sp + parallel
    # tiling) actually has an SP group to split work over. Previously this
    # was hardcoded to (1, 1), so 4-rank preprocess silently degraded to
    # 4-way data parallelism and every rank re-encoded the full clip.
    sp_size = max(1, getattr(fastvideo_args, "sp_size", 1) or 1)
    tp_size = max(1, getattr(fastvideo_args, "tp_size", 1) or 1)
    maybe_init_distributed_environment_and_model_parallel(tp_size, sp_size)
    preprocess_workflow_cls = WorkflowBase.get_workflow_cls(fastvideo_args)
    preprocess_workflow = preprocess_workflow_cls(fastvideo_args)
    preprocess_workflow.run()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    fastvideo_args = FastVideoArgs.from_cli_args(args)
    main(fastvideo_args)
