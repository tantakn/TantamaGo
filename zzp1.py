import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import logging
mylog = logging.getLogger("mylog")
mylog.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)



# import numpy as np
# print("qwer")

# def f (a):
#     return a + 3

# return [self.board[self.get_symmetrical_coordinate(pos, sym)].value for pos in self.onboard_pos]

# board = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# board_data = [0] * 9
# board_data = [1,1,1,1,1,1,1,1,1]
# board_data = [1,1,1,2,1,1,1,0,1]

# board_plane = np.identity(3)[board_data].transpose()

# print(board_plane)
# # for i in range (10):
# #     print(f(i))


import datetime###########
dt_now = datetime.datetime.now()############
for epoch in range(3):
    print(f"sl-model_{dt_now.strftime("%Y%m%d_%H%M%S")}_e:{epoch:0>2}.bin")

print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
print(dt_now.strftime("%Y%m%d_%H%M%S"))