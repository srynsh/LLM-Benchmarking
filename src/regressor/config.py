PV_START = 0.5 # options are 'uniform', float
PVIV_START = 0.5 # options are 'uniform', float
PG_START = 'mean' # options are 'mean', 'uniform', float

NUM_RUNS = 1 # number of runs to min over

ERR_EPSILON = 1e-6 # For adding to start point

### constants


GENS = ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'claude_3_opus', 'gemini-1.5-pro', 'qwen-coder-plus', 'deepseek-chat']

MODELS = [
    'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-4o',
    'claude_3_opus', 'claude_3.5_sonnet',
    'gemini-1.5-flash', 'gemini-1.5-pro',
    'qwen-coder-plus',
    'deepseek-chat'
]

MODEL_NAMES = [
    'GPT 3.5T', 'GPT-4', 'GPT 4o-M', 'GPT 4o',
    'Opus 3', 'Sonnet 3.5',
    'G 1.5 flash', 'G 1.5 pro',
    'Qwen', 'Deepseek'
]

MODEL_ENUM = {model: i for i, model in enumerate(MODELS)}