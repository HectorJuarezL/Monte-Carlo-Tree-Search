import numpy as np
import chess


class simple_agent():
    def get_move_values(self,board,both_players=False):
        moves=list(board.legal_moves)
        values = np.zeros([len(moves),2])
        for i,move in enumerate(moves):
            whites=0
            blacks=0
            b=str(board).replace(' ','').replace('\n','').replace('.','')
            for l in b:
                if l.islower():
                    blacks+=simple_agent.weights[l]
                else:
                    whites+=simple_agent.weights[l]
            suma = whites+blacks
            relacion = whites/suma
            values[i,:]=[relacion,1-relacion]
            #if relacion>0.5:
            #    values[i,:]=[1,0]
            #elif relacion<0.5:
            #    values[i,:]=[0,1]
            #else:
            #    tmp = np.random.randint(2)
            #    values[i,:]=[tmp,(tmp+1)%2]
        if not both_players:
            values = values[:,0] if board.turn else values[:,1]
        return moves,values

    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]

    weights={
        '.':0,
        'p':1,
        'P':1,
        'b':3,
        'B':3,
        'n':3,
        'N':3,
        'r':5,
        'R':5,
        'q':9,
        'Q':9,
        'k':15,
        'K':15
    }

    
class random_agent():
    def get_move_values(self,board,both_players=False):
        moves=list(board.legal_moves)
        values = np.random.rand(len(moves),1)
        if both_players:
            values = np.concatenate((values,1-values),axis=1)
        return moves,values
    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]