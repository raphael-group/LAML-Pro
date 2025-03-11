#ifndef _DIGRAPH_H
#define _DIGRAPH_H

#include <stack>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <utility>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

/*
  Simple adjacency list representation of a DiGraph where the 
  vertices are represented as integers from 0 to N - 1. And
  the vertices have data associated with them.
*/

template <class T>
class vertex {
public:
    int id;
    T data;
    vertex() {};
    vertex(int id, T data) : id(id), data(data) {};
};

template <class T>
class digraph {
private:
    int id_counter = 0;

    std::vector<std::vector<int>> succ;
    std::vector<std::vector<int>> pred;
    std::vector<vertex<T>> vertices;
public:
    // returns id of created vertex
    int add_vertex(T data) {
        vertex<T> v(id_counter, data);
        vertices.push_back(v);
        succ.push_back(std::vector<int>());
        pred.push_back(std::vector<int>());
        id_counter++;
        return v.id;
    }

    void add_edge(int u, int v) {
        succ[u].push_back(v);
        pred[v].push_back(u);
    }

    size_t size() const {
        return vertices.size();
    }

    std::vector<int> nodes() const {
        std::vector<int> vertices;
        for (int i = 0; i < id_counter; i++) {
            vertices.push_back(i);
        }
        return vertices;
    }

    std::vector<std::pair<int, int>> edges() const {
        std::vector<std::pair<int, int>> edges;
        for (size_t u = 0; u < succ.size(); u++) {
            for (const auto &v : succ[u]) {
                edges.push_back(std::make_pair(u, v));
            }
        }
        return edges;
    }

    vertex<T>& operator[](int u) {
        return vertices[u];
    }

    const vertex<T>& operator[](int u) const {
        return vertices[u];
    }

    const std::vector<int>& predecessors(int u) const {
        return pred[u];
    }

    const std::vector<int>& successors(int u) const {
        return succ[u];
    }

    bool contains(int u) const {
        return vertices.find(u) != vertices.end();
    }

    size_t in_degree(int u) const {
        return pred[u].size();
    }

    size_t out_degree(int u) const {
        return succ[u].size();
    }

    std::vector<int> preorder_traversal(int root) {
        std::vector<int> preorder;
        std::stack<int> call_stack;
        call_stack.push(root);
        while(!call_stack.empty()) {
            int i = call_stack.top();
            call_stack.pop();

            preorder.push_back(i);

            const auto& children = this->successors(i);
            for (auto k : children) {
                call_stack.push(k);
            }
        }

        return preorder;
    }

    std::vector<int> postorder_traversal(int root) {
        std::stack<int> call_stack;
        call_stack.push(root);

        std::vector<int> postorder;
        std::vector<bool> visited(id_counter, false);

        while(!call_stack.empty()) {
            int i = call_stack.top();
            call_stack.pop();

            if (visited[i]) continue;

            if (this->out_degree(i) == 0) {
                visited[i] = true;
                postorder.push_back(i);
                continue;
            }

            const auto& children = this->successors(i);
            bool all_children_valid = true; 
            for (auto k : children) {
              if (!visited[k]) {
                if (all_children_valid) {
                  call_stack.push(i);
                }

                call_stack.push(k);
                all_children_valid = false;
              }
            }

            if (!all_children_valid) continue;

            visited[i] = true;
            postorder.push_back(i);
        }

        return postorder;
    }

    /*
     * Returns true if u is an ancestor of v in the given tree.
     */
    friend bool ancestor(const digraph<T>& tree, int u, int v) {
        if (u == v) {
            return true;
        }

        for (int w : tree.succ[u]) {
            if (ancestor(tree, w, v)) {
                return true;
            }
        }

        return false;
    }
};

template <class T>
std::string to_adjacency_list(const digraph<T>& G, const std::unordered_map<int, int>& vertex_map) {
    std::stringstream ss;
    for (const auto& u : G.nodes()) {
        ss << G[u].data.id << " ";
        for (const auto& v : G.successors(u)) {
            ss << G[v].data.id << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

/*
 * Parses an adjacency list into a directed graph object, 
 * where the vertices are read in as integers.
 */
inline std::pair<digraph<int>, std::unordered_map<int, int>> parse_adjacency_list(const std::string& filename) {
    digraph<int> g;

    std::ifstream file(filename);
    std::string line;
    std::unordered_map<int, int> vertex_map;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    while (std::getline(file, line)) {
        if (line.length() < 1 || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int src, tgt;

        if (!(iss >> src)) {
            break;
        }

        if (vertex_map.find(src) == vertex_map.end()) {
            vertex_map[src] = g.add_vertex(src);
        }

        while (iss >> tgt) {
            if (vertex_map.find(tgt) == vertex_map.end()) {
                vertex_map[tgt] = g.add_vertex(tgt);
            }

            g.add_edge(vertex_map[src], vertex_map[tgt]);
        }
    }

    file.close();
    return std::make_pair(g, vertex_map);
}

#endif
