
class Dijkstra:

    def __init__(self, graph, src):
      self.src = src
      self.graph = graph
      self.sptSet = set()     
      self.dist = {}

    
    def compute(self):

      self.dist[self.src] = {}

      for node in self.graph:
        self.dist[self.src][node] = 1e7

      self.dist[self.src][self.src] = 0

      for _ in self.dist[self.src]:
        u = self.minDistance()

        self.sptSet.add(u)
        
        for v in self.graph[u]:
          if (self.graph[u][v] > 0 and
              v not in self.sptSet and
              self.dist[self.src][v] > self.dist[self.src][u] + self.graph[u][v]):
            self.dist[self.src][v] = self.dist[self.src][u] + self.graph[u][v]


      #print(self.dist)

      
    def minDistance(self):
      min = 1e7

      for v in self.dist[self.src]:
        if self.dist[self.src][v] < min and v not in self.sptSet:
          min = self.dist[self.src][v]
          min_v = v
      
      return min_v
