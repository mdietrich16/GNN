import gnn.utils.utils_fast
import gnn.utils.utils_slow
import sys
#sys.modules['gnn.utils.utils'] = sys.modules['gnn.utils.utils_fast']
sys.modules['gnn.utils.utils'] = sys.modules['gnn.utils.utils_slow']
