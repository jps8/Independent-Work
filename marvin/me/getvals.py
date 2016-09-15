import chess, chess.pgn, chess.uci
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import time
import marvin_io

data_num = 1000
data_name = 'first1k'
#initialize uci of the engine!
engine = chess.uci.popen_engine("/Users/joesheehan/documents/iw/stockfish")
engine.uci()
engine.debug(True)
info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

def bb2array(b, flip=False):
    x = numpy.zeros(64, dtype=numpy.float16)
    ls = []
    for pt in range(1,7):
        for pos, piece in enumerate(b.pieces(pt,True)):
            if piece != 0:
                color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
                col = int(pos % 8)
                row = int(pos / 8)
                if flip:
                    row = 7-row
                    color = 1 - color
                piece = color*7 + piece

                x[row * 8 + col] = numpy.float16(piece)
        for pos, piece in enumerate(b.pieces(pt,False)):

            if piece != 0:
                color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
                col = int(pos % 8)
                row = int(pos / 8)
                if flip:
                    row = 7-row
                    color = 1 - color

                piece = color*7 + piece

                x[row * 8 + col] = numpy.float16(piece)
    return x


def parse_game(g):
    rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    r = g.headers['Result']
    if r not in rm:
        return None
    y = rm[r]

# Generate all boards
    gn = g.end()
    bl =[]
    ll =[]

    while gn:
        #calculate score from stockfish
        engine.position(gn.board())
        command = engine.go(async_callback=True, movetime = 1, ponder=False)
        command.result()
        score = info_handler.info["score"][1].cp
        if not score:
            mate = info_handler.info["score"][1].mate
            if mate == 0:
                score = 999999
            elif mate:
                score = (int) ((1. / info_handler.info["score"][1].mate) * 100000)
            else:
                score = 0

        b = gn.board()
        x = bb2array(b)
        score = numpy.float16(score)
        bl.append(x)
        ll.append(score)

        gn = gn.parent


    return (bl,ll)



def samplefunc():
    f = open('/Users/joesheehan/documents/iw/games.pgn')
    bdb = []
    ldb = []
    ts = time.time()


    for n in range(0,data_num):
        if n % 10 == 0:
            ct = time.time() - ts
            print 'n is ', n, ' time is ', ct
        g = chess.pgn.read_game(f)
        agame = parse_game(g)
        if agame is None:
            continue
        bl, ll = agame
        for i in bl:
            bdb.append(i)
        for i in ll:
            ldb.append(i)


    print "done processing"
    t1 = marvin_io.Tensor()
    t2 = marvin_io.Tensor()
    t1.type = 'half';
    t1.sizeof = 2;
    t1.name = 'chess1k train boards';
    t1.value = numpy.array(bdb);
    t1.dim= 4;
    marvin_io.write_tensor('./chess1000b.tensor', t1); 

    t2.type = 'half';
    t2.sizeof = 2;
    t2.name = 'chess1k train labels';
    t2.value = numpy.array(ldb);
    t2.dim= 4;
    marvin_io.write_tensor('./chess1000l.tensor', t2);



if __name__ == '__main__':
    samplefunc()

