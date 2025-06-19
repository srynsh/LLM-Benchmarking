from enum import Enum

# =========================
#   NUMBER OF SIDS
# =========================
NUM_SIDS = 366

# =========================
#   PATHS
# =========================
pathData = './data/'
pathOutput = './output/'
pathLatex = './output/latex/'
pathLogs = './logs/'

pathGenerator = f'{pathData}/generator/'
pathValidator = f'{pathData}/validator/'

pathValidatorOutput = f'./src/validation/generated_scripts'
fpathLLMAsJudge = f'{pathValidatorOutput}/llm_as_judge.py'
fpathValidatorSummary = f'{pathValidatorOutput}/summary.py'

# =========================
#   VALIDATION PATHS
# =========================
VALIDATION_LOGS_DIR = './new_logs'
VALIDATION_OUTPUT_DIR = './validation_output'

# =========================
#   Validator Repair
# =========================
# Feature toggle configuration
class ValidatorRepairConfig:
    def __init__(self):
        # Feedback matching 
        self.feedback_match_fuzzy = True # Fuzzy match for feedback
        self.clip_feedback_lazy = True # Validator got lazy and gave short feedback

        # Line number matching
        self.line_num_number = True # Replace "line-num" with "line number"
        self.line_number_hyphens = True # Line numbers contain hyphens "1-3"

        # Classification labeling
        self.partially_valid_label = True # Validator gave new label "partially valid"
        

# Global instance
VALIDATOR_REPAIR = ValidatorRepairConfig()


# =========================
#   MODEL CONFIGURATION
# =========================

# TODO: Merge these with the model names in the Model enum

# Model retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # seconds

# Supported model names for validation
SUPPORTED_GENERATOR_MODELS = [
    "gpt-4o",
    "gpt-4-turbo", 
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "claude_3_opus",
    "claude_3.5_sonnet",
    "claude_3.5_haiku",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17", 
    "qwen-coder-plus",
    "deepseek-chat"
]

SUPPORTED_VALIDATOR_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo", 
    "gpt-4o-mini",
    "claude_3_opus",
    "claude_3.5_sonnet",
    "claude_3.5_haiku",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17",
    "qwen-coder-plus",
    "deepseek-chat"
]

# =========================
#   MODEL NAMES (Legacy)
# =========================

# Sonnet : anthropic.claude-3-5-sonnet-20241022-v2:0
# Haiku: anthropic.claude-3-5-haiku-20241022-v1:0
# Opus: anthropic.claude-3-opus-20240229-v1:0
# Gemini 1.5 Flash: gemini-1.5-flash-002
# Gemini 1.5 Pro: gemini-1.5-pro-002
# GPT 4: gpt-4-0613
# GPT 4T: gpt-4-turbo-2024-04-09
# GPT 4o: gpt-4o-2024-11-20
# GPT 4o mini: gpt-4o-mini-2024-07-18
# Qwen: qwen-coder-plus-2024-11-06
# Deepseek: deepseek-chat 

# class Model(Enum):
#     GPT_3_5_TURBO = "gpt-3.5-turbo"
#     GPT_4_TURBO = "gpt-4-turbo-2024-04-09"
#     GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
#     GPT_4O = "gpt-4o-2024-08-06"
#     GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
#     GPT_4_1 = "gpt-4.1-2025-04-14"
#     CLAUDE_3_OPUS = "claude_3_opus"
#     CLAUDE_3_5_SONNET = "claude_3.5_sonnet"
#     CLAUDE_3_5_HAIKU = "claude_3.5_haiku"
#     GEMINI_1_5_FLASH = "gemini-1.5-flash"
#     GEMINI_1_5_PRO = "gemini-1.5-pro"
#     GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-04-17"
#     GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"
#     QWEN_CODER_PLUS = "qwen-coder-plus"
#     DEEPSEEK_CHAT = "deepseek-chat"


class Model(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1 = "gpt-4.1"
    CLAUDE_3_OPUS = "claude_3_opus"
    CLAUDE_3_5_SONNET = "claude_3.5_sonnet"
    CLAUDE_3_5_HAIKU = "claude_3.5_haiku"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    QWEN_CODER_PLUS = "qwen-coder-plus"
    DEEPSEEK_CHAT = "deepseek-chat"


# List of all model values
MODELS = [
    Model.GPT_3_5_TURBO.value,
    Model.GPT_4_TURBO.value,
    Model.GPT_4O_MINI.value,
    Model.GPT_4O.value,
    Model.GPT_4_1_MINI.value,
    Model.GPT_4_1.value,
    Model.CLAUDE_3_OPUS.value,
    Model.CLAUDE_3_5_SONNET.value,
    Model.CLAUDE_3_5_HAIKU.value,
    Model.GEMINI_1_5_FLASH.value,
    Model.GEMINI_1_5_PRO.value,
    Model.GEMINI_2_5_FLASH.value,
    Model.GEMINI_2_5_PRO.value,
    Model.QWEN_CODER_PLUS.value,
    Model.DEEPSEEK_CHAT.value
]

MODELS_GEN = [
    Model.GPT_4_TURBO.value,
    Model.GPT_4O.value,
    Model.CLAUDE_3_OPUS.value,
    Model.GEMINI_1_5_PRO.value,
    Model.QWEN_CODER_PLUS.value,
    Model.DEEPSEEK_CHAT.value
]

mapping_latex = {
    Model.GPT_3_5_TURBO.value: '\\gptThreeTurbo',
    Model.GPT_4_TURBO.value: '\\gptFour',
    Model.GPT_4O_MINI.value: '\\gptFourOMini',
    Model.GPT_4O.value: '\\gptFourO',
    Model.CLAUDE_3_OPUS.value: '\\opus',
    Model.CLAUDE_3_5_SONNET.value: '\\sonnet',
    Model.CLAUDE_3_5_HAIKU.value: '\\haiku',
    Model.GEMINI_1_5_FLASH.value: '\\flash',
    Model.GEMINI_1_5_PRO.value: '\\pro',
    Model.QWEN_CODER_PLUS.value: '\\qwen',
    Model.DEEPSEEK_CHAT.value: '\\deepseek',
    Model.GEMINI_2_5_FLASH.value: '\\flashTwoFive',
    Model.GEMINI_2_5_PRO.value: '\\proTwoFive',
    Model.GPT_4_1.value: '\\gptFourOne',
    Model.GPT_4_1_MINI.value: '\\gptFourOneMini'
}

# =========================
#   MODEL SETTINGS
# =========================
MAX_RETRY_ATTEMPTS = 3  # Number of retries for failed JSON parsing
RETRY_DELAY = 1  # Delay between retries in seconds

# =========================
#   ENSEMBLE CONFIG
# =========================
VALID_K = range(1, 11)
INVALID_K = range(1, 11)

# =========================
#   GOLD CONFIG
# =========================
pathGold = f'{pathData}/generator/'
fnameGoldTable = f'{pathLatex}/table_precision.tex'

# =========================
#   VALIDATOR CONFIG
# =========================
pathValidator = f'{pathData}/validator/'
tpr_tnr_genModel = ['claude_3_opus']

# =========================
#   LATEX CONFIG
# =========================
fnamePredMatrix = f'{pathLatex}/table_predicted_matrix.tex'
fnameTprTnrTex = f'{pathLatex}/table_tpr_tnr.tex'