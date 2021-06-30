import chess
import numpy as np
import time

class MCTSNode:
    def __init__(self, game_state, parent = None, move = None, value = [0.5,0.5], bot = None, isRoot = False ):
        self.game_state = game_state #estado de juego
        self.parent = parent #nodo padre
        self.move = move #movimiento asociado al nodo
        self.Q = np.zeros([2]) #valor de calidad del nodo
        self.value=value #valor de la función de evaluación
        self.N = 0 #Numero de veces visitado
        self.children = [] #lista de nodos hijos
        self.unvisited_moves,self.unvisited_values = bot.get_move_values(game_state,both_players=True) #movimientos no visitados con sus respectivos valores
        if type(self.unvisited_values) is np.ndarray: #conversión de tipos de dato en caso de recibir variables de numpy
            self.unvisited_values = self.unvisited_values.tolist()
        self.isRoot=isRoot #booleano para comprobar si es un nodo raiz

    def add_random_child(self,bot):
        index = np.random.randint(len(self.unvisited_moves))
        new_game_state = self.game_state.copy()
        new_move = self.unvisited_moves.pop(index)#selecciona un movimiento disponible al azar y lo elimina de los movimientos no visitados
        new_value = self.unvisited_values.pop(index)

        new_game_state.push(new_move) #realiza el movimiento seleccionado
        new_node = MCTSNode(game_state=new_game_state, parent=self, move=new_move,value=new_value,bot=bot) #crea un nuevo nodo
        self.children.append(new_node) #añade el nodo a su lista de hijos
        return new_node #retorna el nuevo nodo

    def update_q(self, result):
        self.Q += result
        self.N += 1

    def can_add_child(self): #comprueba si aun hay nodos por visitar
        return len(self.unvisited_moves) > 0

    def is_terminal(self): #verifica si es un nodo terminal, es decir, el final de una partida
        return self.game_state.is_game_over()

    def Q_frac(self, player): #obtiene el valor Q/N para el nodo dado
        if player: #turno de las blancas
            return float(self.Q[0]) / float(self.N)
        else: #turno de las negras
            return float(self.Q[1]) / float(self.N)

class MCTS:
    def __init__(self, temperature=None,bot=None,game_state=None,default_time=30):
        self.temperature = temperature
        self.bot = bot
        self.root = MCTSNode(game_state,bot=self.bot,isRoot=True)
        self.default_time=default_time

    def select_move(self, game_state,time_len=None):
        
        if self.root.game_state!=game_state:
            print('\nEl estado de juego no corresponde con el de la raiz del arbol, se recreó la raiz')
            self.root = MCTSNode(game_state,bot=self.bot,isRoot=True)

        root=self.root
        if time_len is None:
            time_len = self.default_time
        #print("\n")
        #i=0
        start=time.time()
        while time.time()-start<time_len:
            #print(i,end=" ")
            #i+=1
            node = root
            #fase de seleccion, donde busca un nodo que no sea un nodo derminal
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            #fase de expansión, donde se agrega un nuevo nodo
            if node.can_add_child():
                node = node.add_random_child(self.bot)

            #fase de simulación. Con ayuda de la red neuronal, se obtiene el valor del nodo que predice como ganador
            winner = node.value

            #fase de retropropagación, donde se actualiza el valor de Q de los nodos padres hasta llegar al nodo raiz
            while node is not None:
                node.update_q(winner)
                node = node.parent

        #a continuación, se crea una lista de set donde se recupera la información de cada uno de los hijos del nodo raiz con sus respectivos valores Q y N
        scored_moves = [
            (child.Q_frac(root.game_state.turn), child.move, child.N)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        if len(scored_moves)>0:
            best_move=scored_moves[0][1]
            print('\nSelect move %s with win pct %.3f' % (best_move, scored_moves[0][0]))
            
            self.push_move(best_move)
            return best_move
        else:
            print('posible error o nodo terminal')
            return None
        
    def push_move(self,move):
        root=self.root
        for child in root.children:
            if child.move==move:
                child.isRoot=True
                self.root=child
                return
        print("Error, movimiento no existente")
        return

    def select_child(self, node):
        """
            Selecciona un hijo usando la métrica UCT (Upper confidence bound for trees).
        """
        #Calcula N(v)
        total_rollouts = sum(child.N for child in node.children)
        log_rollouts = np.log(total_rollouts)

        best_score = -1
        best_child = None
        #Calcula UTC(j)
        for child in node.children:
            win_percentage = child.Q_frac(node.game_state.turn)
            exploration_factor = np.sqrt(log_rollouts / child.N)
            uct_score = win_percentage + self.temperature * exploration_factor
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
    