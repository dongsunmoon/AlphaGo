import numpy as np
import re
import os
import cPickle as pickle

class SgfReader:
    def __init__(self):
        self.write_file_prefix = "./out/data-"
        self.write_file_extend = ".pkl"
        self.data_array = np.zeros([0,361])
        self.label_array = np.zeros(0)
        self.MAX_DATA_COUNT_IN_ONE_FILE = 1000
        self.read_file_count = 0
        self.write_file_count = 0
        self.total_rollout_count = 0

    def print_board(self, board):
        print 'a b c d e f g h i j k l m n o p q r s'
        for row in board:
            for e in row:
                if e == 0:
                    print '_',
                elif e == 1:
                    print '*',
                else:
                    print 'o',
            print ''

    def insert_board_to_array(self, board2d, stone):
        self.total_rollout_count += 1
        board1dCopy = np.copy(board2d.flatten())
        self.data_array = np.vstack([self.data_array, board1dCopy])

    def save_board_if_read_enough(self):
        if self.total_rollout_count % self.MAX_DATA_COUNT_IN_ONE_FILE != 0:
            return
        self.save_pkl()
        self.flush_buffer()

    def save_pkl(self):
        self.write_file_count += 1
        filename = self.write_file_prefix + str(self.write`file_count) + self.write_file_extend
        pickle.dump( self.data_array, open( filename, "wb" ) )
        print "* write   : %s " % filename
        print "  rollout : %d " % self.total_rollout_count

    def flush_buffer(self):
        self.data_array = np.zeros([0,361])
        self.label_array = np.zeros(0)

    def read_sgf(self, fullpath):
        self.read_file_count += 1
        print 'file%s: %s' % (self.read_file_count , fullpath)
        board = np.zeros((19, 19))
        f = open(fullpath, 'r')

        data = f.read()
        data = data[1:-2]
        # data = data.split(';')

        data = re.split('[\;\]]',data)

        for e in data:
            e = e.strip()
            if len(e) == 0:
                continue
            (key, value) = e.split("[")
            # print '>>'+key+'->'+value

            if key == "SZ":
                # print "size: " + value
                if int(value) != 19:
                    break
            if key in ['B', 'W']:
                if len(value) == 0:
                    continue
                x = ord(value[0])-ord('a')
                y = ord(value[1])-ord('a')
                if key == 'B':
                    board[y][x] = 1;
                else:
                    board[y][x] = -1;
                self.insert_board_to_array(board, key)
                self.save_board_if_read_enough()
                # self.print_board(board)
        f.close()        


if __name__ == "__main__":
    reader = SgfReader()
    for root, dirs, files in os.walk('./KGS_to_201505'):
        for fname in files:
            if fname.endswith('.sgf'):
                fullpath = os.path.join(root, fname)
                reader.read_sgf(fullpath)

    reader.save_pkl()