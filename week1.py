graph = {
 '5' : ['3','7'],
 '3' : ['2','4'],
 '7' : ['8'],
 '2' : [ ],
 '4' : ['8'],
 '8' : [ ],
}
visited = []
queue = []
def bfs(visited,graph,node):
 visited.append(node)
 queue.append(node)
 while queue:
  m = queue.pop(0)
  print(m, end = " ")
  for neighbour in graph[m]:
   if neighbour not in visited:
    visited.append(neighbour)
    queue.append(neighbour)

print("following is the breadth-first")
bfs(visited, graph, '5')

graph = {
 'A': ['S', 'B', 'C'],
 'B': ['A', 'D', 'E'],
 'C': ['A', 'G'],
 'D': ['B'],
 'S': ['A', 'H'],
 'E': ['B'],
 'H': ['S', 'I', 'J'],
 'I': ['H', 'K'],
 'J': ['H'],
 'K': ['I'],
 'G': ['C'],
}
visited = set()
def DFS(node, visited, graph):
 if node not in visited:
  print(node)
  visited.add(node)
  for i in graph[node]:
   DFS(i, visited, graph)

print("following is the DFS")
DFS("A", visited, graph)
