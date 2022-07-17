
from math import ceil
from abc import ABC, abstractmethod
from keras.layers import Layer


class ICellularAutomata(ABC):

    """
    An abstract base class providing template to create a cellular automata. 
    The cellular automata created using this template can have atmost one child
    and one parent cellular automata

    Attributes
    ----------
    _parent : ICellularAutomata
        Immediate parent of the current CA
    _child: ICellularAutomata
        Immediate child of the current CA
    _level : int
        Level of CA in the hierarchy. Root CA is level 0. Level of the CA also keeps track of the 
        number of signals that it can expect from the higher order CAs.
    _cell_states: object
        This includes cell states (self state + signals) and cell connectivity information.
        Its data structure varies based on the nature of the CA ( leaf, HCA )
        Hence the implementation is left to the derived class 
    _parent_ca_cell_ids: object
        The cell id of the parent CA to which each cell of the current CA belongs
        Its data structure varies based on the nature of the CA ( leaf, HCA )
        Hence the implementation is left to the derived class 


    Methods
    -------
    add_child_ca( self, ca)
        Adds a 'ca' object as a child to the current cellular automata object
    remove_child_ca( self )
        Removes link of the current CA with its child
    process_signal(self, signal, parent_cell_id ):
        updates the signal channels of the cell with value parent_cell_id with 'signal' parameter.  
    update_ca(self, make_recursive=False )  
        Updates the CA cells and their neighbors based on the new CA state.
        If make_recursive parameter is set to True,
        the parent CAs are also updated after the current CA is updated.  
    """
    def __init__(self):
        
        self._parent =  None
        self._child  = None
        self._level = 0
        #self._cell_states = None
        #self._parent_ca_cell_ids = None
        
 
    @property    
    def parent(self):
        return self._parent

    @parent.setter
    def parent( self, p):
        self._parent = p

    
    @property    
    def child(self):
        return self._child
    
    @property    
    def level(self):
        return self._level

    # @property
    # def cell_states(self):
    #     return self._cell_states

    
    # @abstractmethod
    # @cell_states.setter
    # def cell_states( self, states):
    #     pass

    # @property
    # def parent_ca_cell_ids(self):
    #     return self._parent_ca_cell_ids   

    # @abstractmethod
    # @parent_ca_cell_ids.setter
    # def parent_ca_cell_ids( self, ids):
    #     pass

    def add_child_ca( self, ca):
        
        ca.parent = self
        self._child =  ca

        node = self
        while node is not None:
            node._child._level  = node._level + 1 
            node = node._child

        
    def remove_child_ca( self ):
        if self._child is not None:
            del self._child
            self._child = None

            # removing the child deletes the complete object hierarchy
            # below it. This is done for simplicity and can be enhanced in future 
            
   # @abstractmethod
    def process_signal(self, signal, parent_cell_id ):
        pass

    # @abstractmethod
    def update_ca(self, make_recursive=False ):
        # Update self._cell_states
        # Update the self._child._parent_ca_cell_ids of the child CAs
        pass