from gerrychain import (GeographicPartition, Partition)

class HashAssignment:
    def __init__(self, prime):
        """
        Param: prime: prime number used in hashing function. It should minimize the chance of a collision.
        """
        self.prime = prime
        self.value = 0

    def __call__(self,partition):
        if partition.parent is None: #first step in the chain
            return self._initialize_hash(partition)
        return self._update_hash(partition)

    def _initialize_hash(self,partition):
        for node, part in partition.assignment.items():
            self.value += 3*(part+1)**node
        return self.value % self.prime

    def _update_hash(self,partition):
        flips = partition.flips #set of nodes that were fliped and their new assignment
        parent = partition.parent
        for node,part in flips.items():
            old_assignment = parent.assignment.to_dict()[node]
            out_flow = (3*(old_assignment+1)**node)
            in_flow = (3*(part+1)**node)
            self.value = (self.value - out_flow + in_flow)%self.prime
        return self.value
