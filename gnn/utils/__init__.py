import utils.utils_fast
import utils.utils_slow
import sys
sys.modules['utils.utils'] = sys.modules['utils.utils_fast']
# sys.modules['utils.utils'] = sys.modules['utils.utils_slow']
