
class TaskType:
    """
    TaskType class is used to define the type of task names store in session.
    Don't use Enum here to avoid json marshalling issues.
    """
    voice_clone = "voice_clone"
    train_sovits = "train_sovits"
    train_gpt = "train_gpt"


class TaskCMD:
    tran_sovits = "train_sovits.py"
    train_gpt = "train_gpt.py"
