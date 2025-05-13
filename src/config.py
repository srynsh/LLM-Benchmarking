# =========================
#   PATHS
# =========================
pathData = '../data/'
pathOutput = '../output/'
pathLatex = '../output/latex/'

# =========================
#   MODEL NAMES
# =========================

MODELS = ['gpt-4-turbo', 'gpt-4o-mini', 'gpt-4o', 
          'claude_3_opus', 'claude_3.5_sonnet', 'gemini-1.5-flash', 'gemini-1.5-pro', 'qwen-coder-plus', 'deepseek-chat', 
          'claude_3.5_haiku', 'gemini-2.5-flash-preview-04-17', 'gemini-2.5-pro-preview-03-25', 'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14']
MODELS_GEN = ['gpt-4-turbo', 'gpt-4o', 'claude_3_opus', 'gemini-1.5-pro', 'qwen-coder-plus', 'deepseek-chat']

mapping = {
    'gpt-3.5-turbo': '\\gptThreeTurbo',
    'gpt-4-turbo': '\\gptFour',
    'gpt-4o-mini': '\\gptFourOMini',
    'gpt-4o': '\\gptFourO',
    'claude_3_opus': '\\opus',
    'claude_3.5_sonnet': '\\sonnet',
    'gemini-1.5-flash': '\\flash',
    'gemini-1.5-pro': '\\pro',
    'qwen-coder-plus': '\\qwen',
    'deepseek-chat': '\\deepseek',
    'claude_3.5_haiku': '\\haiku', 
    'gemini-2.5-flash-preview-04-17': '\\flashTwoFive', 
    'gemini-2.5-pro-preview-03-25': '\\proTwoFive', 
    'gpt-4.1-2025-04-14': '\\gptFourOne', 
    'gpt-4.1-mini-2025-04-14': '\\gptFourOneMini'
}

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