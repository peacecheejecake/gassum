class Node:
    def __init__(self, char, prev=None, post=None):
        self.char = char
        self.prev = prev
        self.post = post


def solution(s):
    nodes = [Node(char=char) for char in s]
    nodes[0].post = nodes[1]
    nodes[-1].prev = nodes[-2]
    for i, node in enumerate(nodes[1:-1]):
        node.prev = nodes[i - 1]
        node.post = nodes[i + 1]
    
    first, last = nodes[0], nodes[-1]
    length = len(s)
    node = first
    while length > 0:
        if node == last:
            return 0

        post = node.post
        if node.char == post.char:
            if node == first:
                first = post.post
            else:
                node.prev.post = post.post
                
            if post.post == last:
                last = node.prev
            else:
                post.post.prev = node.prev
                
            length -= 2
            node = first
        else:
            node = post
    return 1
            
print(solution('cdcd'))