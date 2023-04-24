import cv2
import numpy as np


class UbongoSolver(object):

    def __init__(self):
        pass

    @staticmethod
    def findUbongoSolution(field, blocks):

        if field is None or field.sum() == 0 or len(blocks) == 0 or blocks[0].sum() == 0 :
            return False, []

        cb = blocks[0]
        b = np.zeros_like(field)
        sy, sx = cb.shape[:2]

        try:
            b[:sy, :sx] = cb[:, :]
        except ValueError:
            return False, []

        # check if block is mirror symmetrical
        rotations = 2 if np.all(cb - cv2.flip(cb, -1) == 0) else 4

        for _i in range(rotations):
            while True:
                newField = field-b
                if np.all(newField > -1):
                    # valid block Position found

                    if (newField.sum() + len(blocks)-1) == 0:
                        return True, [b]

                    isValid, blockPositions = UbongoSolver.findUbongoSolution(newField, blocks[1:])

                    if isValid:
                        blockPositions.append(b)
                        return isValid, blockPositions

                #move block
                if np.any(b[..., -1] == 1) and np.any(b[-1, ...] == 1):
                    # all position with this orientation has been tested
                    break
                elif np.any(b[..., -1] == 1):
                    b = np.roll(b, cb.shape[1], axis=1)
                    b = np.roll(b, 1, axis=0)
                else:
                    b = np.roll(b, 1, axis=1)

            #rotate block
            cb = np.rot90(cb)
            b = np.zeros_like(field)
            sy, sx = cb.shape[:2]

            try:
                b[:sy, :sx] = cb[:, :]
            except ValueError:
                return False, []

        return False, []

