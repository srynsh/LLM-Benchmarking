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

pathDataGAIED = f'{pathData}/GAIED/'
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

# =========================
#   MODEL NAMES (Legacy)
# =========================

class ModelVersion(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-0613"
    GPT_4_TURBO = "gpt-4-turbo-2024-04-09"
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
    GPT_4O = "gpt-4o-2024-11-20"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_1 = "gpt-4.1-2025-04-14"
    CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v2:0"
    GEMINI_1_5_FLASH = "gemini-1.5-flash-002"
    GEMINI_1_5_PRO = "gemini-1.5-pro-002"
    GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-04-17"
    GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"
    QWEN_CODER_PLUS = "qwen-coder-plus-2024-11-06"
    DEEPSEEK_CHAT = "deepseek-chat"


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
MODELS_GEN = [
    Model.GPT_4O.value,
    Model.GPT_4_TURBO.value,
    Model.CLAUDE_3_OPUS.value,
    Model.GEMINI_1_5_PRO.value,
    Model.QWEN_CODER_PLUS.value,
    Model.DEEPSEEK_CHAT.value
]

MODELS_VAL = [
        Model.GPT_4_TURBO.value, Model.GPT_4O_MINI.value, Model.GPT_4O.value,
        Model.CLAUDE_3_OPUS.value, Model.CLAUDE_3_5_SONNET.value,
        Model.GEMINI_1_5_FLASH.value, Model.GEMINI_1_5_PRO.value,
        Model.QWEN_CODER_PLUS.value,
        Model.DEEPSEEK_CHAT.value,
        Model.CLAUDE_3_5_HAIKU.value, Model.GEMINI_2_5_FLASH.value, Model.GEMINI_2_5_PRO.value, 
        Model.GPT_4_1.value, Model.GPT_4_1_MINI.value
    ]


MODELS_NAME = [
    'GPT-4', 'GPT 4o-M', 'GPT 4o',
    'Opus 3', 'Sonnet 3.5',
    'G 1.5 flash', 'G 1.5 pro',
    'Qwen', 'Deepseek',
    'Haiku 3.5', 'G 2.5 flash', 'G 2.5 pro', 
    'GPT 4.1', 'GPT 4.1-M'
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
#   GENERATOR CONSTANTS
# =========================

# TODO: Update all the numbers below
pGa_CONST = {
    'gpt-4o': 0.93478,
    'gpt-4-turbo': 0.872,
    'claude_3_opus': 0.95402,
    'gemini-1.5-pro': 0.92846,
    'deepseek-chat': 0.92841,
    'qwen-coder-plus': 0.93117
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