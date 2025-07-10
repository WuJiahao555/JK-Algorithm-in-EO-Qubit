from CCNOT_Shorten import Krotov_optimizer_CCNOT_Shorten
import collections

FW = Krotov_optimizer_CCNOT_Shorten()
FW.mute_initial(mode=1)

counter = collections.Counter(FW.mute_memory.flatten())
print(f"{counter[0]-28} deleted, {counter[1]} not deleted.")
