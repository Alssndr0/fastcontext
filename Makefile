# FastContext convenience commands. These are thin aliases — all serving logic
# (fp8 defaults, context sizing, tool-call parser) lives in serving/serve_vllm.sh,
# and per-box tuning lives in serving/serve.local.env (gitignored).
#
#   make serve         # start the vLLM server (CUDA, fp8)
#   make serve-mlx     # start the MLX server (Apple Silicon)
#   make verify        # check the server is up and tool-calling works
#   make explore Q="find the auth middleware"
#
# Inline overrides still work, e.g.: make serve MAX_MODEL_LEN=140000

.PHONY: serve serve-mlx verify explore

serve:        ## Launch the vLLM server (CUDA, fp8)
	./serving/serve_vllm.sh

serve-mlx:    ## Launch the MLX server (Apple Silicon)
	./serving/serve_mlx.sh

verify:       ## Check the server is up and tool-calling works
	./serving/verify.sh

explore:      ## Run a query: make explore Q="..."
	fastcontext -q "$(Q)" --citation
