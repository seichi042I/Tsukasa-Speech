# Compatibility shim — real code lives in tsukasa_speech.training.stage2
from tsukasa_speech.training.stage2 import *

if __name__ == '__main__':
    from tsukasa_speech.training.stage2 import main
    main()
