import unittest
import numpy as np
from environment import FrozenLake


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.small_lake = np.load('p.npy')
        self.fl = FrozenLake(lake=[['&', '.', '.', '.'],
			['.', '#', '.', '#'],
			['.', '.', '.', '#'],
			['#', '.', '.', '$']], slip=0.1, max_steps=10)

    def test_tiles(self):
        print('test_tiles')
        self.assertEqual(self.fl.lake_flat[0],'&')
        self.assertEqual(self.fl.lake_flat[5], '#')
        self.assertEqual(self.fl.lake_flat[7], '#')
        self.assertEqual(self.fl.lake_flat[11], '#')
        self.assertEqual(self.fl.lake_flat[12], '#')
        self.assertEqual(self.fl.lake_flat[15], '$')

    def test_probabilities(self):
        map_size = len(self.small_lake[0])
        for i in range(map_size):
            for j in range(map_size):
                print('i={} j={}'.format(i, j))
                self.assertEqual(self.fl.p(i, j, 0), self.small_lake[i][j][0])
                self.assertEqual(self.fl.p(i, j, 1), self.small_lake[i][j][1])
                self.assertEqual(self.fl.p(i, j, 2), self.small_lake[i][j][2])
                self.assertEqual(self.fl.p(i, j, 3), self.small_lake[i][j][3])






if __name__ == '__main__':
    unittest.main()
