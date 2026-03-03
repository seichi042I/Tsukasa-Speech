# Compatibility shim — real code lives in tsukasa_speech.training.stage1
from tsukasa_speech.training.stage1 import *

if __name__ == '__main__':
    from tsukasa_speech.training.stage1 import main
    main()
