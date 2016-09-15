import chess, chess.pgn, chess.uci
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import time
import marvin_io

if len(sys.argv) != 3:
    print "defaulting to 1000 10"
    data_num = 1000
    bin_num = 10

else:
    data_num = int(sys.argv[1])
    bin_num = int(sys.argv[2])

#initialize uci of the engine!
engine = chess.uci.popen_engine("/home/jps8/stockfish_dir/stockfish-7-linux/Linux/sfish")
engine.uci()
engine.debug(True)
info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

def bb2array(b, flip=False):
    x = numpy.zeros((8,8), dtype=numpy.float32)
    for row in range(1,9):
        for col in range(1,9):
            cur = b.piece_at((row-1)*8+col-1)
            if cur:
                if cur.color:
                    pieceval = cur.piece_type
                else:
                    pieceval  = cur.piece_type+6
                x[row-1, col-1] = pieceval
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
            if mate > 0:
                score = 99999
            elif mate <= 0:
                score = 88888
            else:
                score = 0

        b = gn.board()
        x = bb2array(b)
        score = numpy.float32(score)
        if b.turn:
            score = -score
        bl.append(x)
        ll.append(score)

        gn = gn.parent


    return (bl,ll)



def samplefunc():
    f = open('/home/jps8/datagen/onegame.pgn')
    b1 = []
    b2 = []
    l1 = []
    l2 = []
    lnet = []
    bindivs = []

    ts = time.time()
    train_boards = 0
    test_boards = 0
    train_games = (3 *data_num) // 4
    test_games = data_num-train_games

    didend = 0
    for n in range(0,train_games):
        if n % 10 == 0:
            ct = time.time() - ts
            print 'n is ', n, ' time is ', ct
        g = chess.pgn.read_game(f)
        if g is None:
            didend = n
            break
        agame = parse_game(g)
        if agame is None:
            continue
        bl, ll = agame
        for i in bl:
            train_boards = train_boards+1
            b1.append(i)
        for i in ll:
            l1.append(i)
            lnet.append(i)

    print "on training data"
    if didend != 0:
        print "ended on", didend
    else:
        for n in range(0,test_games):
            if n % 10 == 0:
                ct = time.time() - ts
                print 'n is ', n+train_games, ' time is ', ct
                g = chess.pgn.read_game(f)
                if g is None:
                    didend = -n
                    break
                agame = parse_game(g)
                if agame is None:
                    continue
                bl, ll = agame
                for i in bl:
                    test_boards = test_boards+1
                    b2.append(i)
                for i in ll:
                    l2.append(i)
                    lnet.append(i)

    print "didend = ", didend
    print "done processing"
    lnet.sort()
    binsize = len(lnet) // bin_num
    numbigger = len(lnet) % bin_num
    sofar = 0
    for n in range(0, bin_num-1):
        if n < numbigger:
            sofar += binsize + 1
        else:
            sofar += binsize
        curdiv = lnet[sofar]
        if (curdiv!= 88888) and (curdiv != 99999):
            bindivs.append(curdiv)

    bindivs.append(88880)
    bindivs.append(99990)

    l1 = numpy.digitize(numpy.array(l1), bindivs)
    l2 = numpy.digitize(numpy.array(l2), bindivs)
   
    print "train boards ", train_boards
    print "b1 is ", len(l1)
    b1 = numpy.array(b1).reshape(train_boards, 1, 8, 8)
    b2 = numpy.array(b2).reshape(test_boards, 1, 8, 8)

    print "test boards ", test_boards
    l1 = l1.reshape(train_boards,1, 1, 1)
    l2 = l2.reshape(test_boards,1, 1, 1)

    t1 = marvin_io.Tensor()
    t2 = marvin_io.Tensor()
    t3 = marvin_io.Tensor()
    t4 = marvin_io.Tensor()

    special =  str(data_num) + "_"+str(bin_num)
    name1 =str("./b1_" + special + (".tensor"))
    t1.name = 'boards train '+ special
    t1.value = b1.astype(numpy.float32)
    marvin_io.write_tensor(name1, t1)
    testt(name1)

    t2.name = 'boards test '+ special
    t2.value = b2.astype(numpy.float32)  
    name2 =str("./b2_" + special + (".tensor"))
    marvin_io.write_tensor(name2, t2)
    testt(name2)

    t3.name = 'labels train '+ special
    t3.value = l1.astype(numpy.float32)
    name3 =str("./l1_" + special + (".tensor"))
    marvin_io.write_tensor(name3, t3)
    testt(name3)

    t4.name = 'labels test '+ special
    t4.value = l2.astype(numpy.float32)
    name4 =str("./l2_" + special + (".tensor"))
    marvin_io.write_tensor(name4, t4)
    testt(name4)



def testt(tensname):
    tl = marvin_io.read_tensor(tensname)
    for t in tl:
        print t
        #print 'type ', t.type
        #print 'sizeof ', t.sizeof
        print 'name ', t.name
        #print 'value ', t.value
        #print 'dim ', t.dim



if __name__ == '__main__':
    samplefunc()

