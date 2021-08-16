# Copyright (c) 2020-present, Hao BAI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def is_directional(action, choice):
    print(choice)
    if (action == choice) or (action == choice + " 1") or (action == choice + " 2") or (action == choice + " 3"):
        return True
    return False
