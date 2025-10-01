from QFT3_Shorten import Krotov_optimizer_QFT3_Shorten
import collections

FW = Krotov_optimizer_QFT3_Shorten()
FW.mute_initial(mode=1)
print(FW.mute_memory)

counter = collections.Counter(FW.mute_memory.flatten())
print(f"{counter[0]} deleted, {counter[1]} not deleted.")
