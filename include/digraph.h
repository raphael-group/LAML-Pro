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

template <class T>
class vertex {
public:
    int id;
    T data;
    vertex() {};
    vertex(int id, T data) : id(id), data(data) {};
};

/**
 * @brief An adjacency list representation of a directed graph.
 * 
 * The vertices are represented as integers from 0 to N-1, with
 * each vertex having arbitrary data (of type T) associated with it.
 * 
 * @tparam T The type of data stored in each vertex.
 */
template <class T>
class digraph {
private:
    int id_counter = 0;

    std::vector<std::vector<int>> succ;
    std::vector<std::vector<int>> pred;
    std::vector<vertex<T>> vertices;
public:
    /**
     * @brief Creates a new vertex in the graph.
     * 
     * @param data The data to store in the new vertex.
     * @return int The ID of the newly created vertex.
     */
    int add_vertex(T data) {
        vertex<T> v(id_counter, data);
        vertices.push_back(v);
        succ.push_back(std::vector<int>());
        pred.push_back(std::vector<int>());
        id_counter++;
        return v.id;
    }

    /**
     * @brief Adds a directed edge from vertex u to vertex v.
     * 
     * @param u The ID of the source vertex.
     * @param v The ID of the destination vertex.
     */
    void add_edge(int u, int v) {
        succ[u].push_back(v);
        pred[v].push_back(u);
    }

    /**
     * @brief Returns the number of vertices in the graph.
     * 
     * @return size_t The number of vertices.
     */
    size_t size() const {
        return vertices.size();
    }

    /**
     * @brief Returns a vector of all vertex IDs in the graph.
     * 
     * @return std::vector<int> A vector containing all vertex IDs.
     */
    std::vector<int> nodes() const {
        std::vector<int> vertices;
        for (int i = 0; i < id_counter; i++) {
            vertices.push_back(i);
        }
        return vertices;
    }

    /**
     * @brief Returns all edges in the graph as pairs of vertex IDs.
     * 
     * @return std::vector<std::pair<int, int>> A vector of (source, destination) vertex ID pairs.
     */
    std::vector<std::pair<int, int>> edges() const {
        std::vector<std::pair<int, int>> edges;
        for (size_t u = 0; u < succ.size(); u++) {
            for (const auto &v : succ[u]) {
                edges.push_back(std::make_pair(u, v));
            }
        }
        return edges;
    }

     /**
     * @brief Access a vertex by its ID.
     * 
     * @param u The ID of the vertex to access.
     * @return vertex<T>& A reference to the vertex.
     */
    vertex<T>& operator[](int u) {
        return vertices[u];
    }

    /**
     * @brief Access a vertex by its ID (const version).
     * 
     * @param u The ID of the vertex to access.
     * @return const vertex<T>& A const reference to the vertex.
     */
    const vertex<T>& operator[](int u) const {
        return vertices[u];
    }

    /**
     * @brief Returns the predecessors (in-neighbors) of a vertex.
     * 
     * @param u The ID of the vertex.
     * @return const std::vector<int>& A vector of vertex IDs that have edges to u.
     */
    const std::vector<int>& predecessors(int u) const {
        return pred[u];
    }

    /**
     * @brief Returns the successors (out-neighbors) of a vertex.
     * 
     * @param u The ID of the vertex.
     * @return const std::vector<int>& A vector of vertex IDs that u has edges to.
     */
    const std::vector<int>& successors(int u) const {
        return succ[u];
    }

    /**
     * @brief Checks if a vertex with the given ID exists in the graph.
     * 
     * @param u The ID to check.
     * @return bool True if the vertex exists, false otherwise.
     */
    bool contains(int u) const {
        return vertices.find(u) != vertices.end();
    }

    /**
     * @brief Returns the in-degree of a vertex (number of incoming edges).
     * 
     * @param u The ID of the vertex.
     * @return size_t The in-degree of the vertex.
     */
    size_t in_degree(int u) const {
        return pred[u].size();
    }

     /**
     * @brief Returns the out-degree of a vertex (number of outgoing edges).
     * 
     * @param u The ID of the vertex.
     * @return size_t The out-degree of the vertex.
     */
    size_t out_degree(int u) const {
        return succ[u].size();
    }

    /**
     * @brief Performs a preorder traversal of the graph starting from the given root.
     * 
     * @param root The ID of the starting vertex.
     * @return std::vector<int> A vector of vertex IDs in preorder.
     */
    std::vector<int> preorder_traversal(int root) const {
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

    /**
     * @brief Performs a postorder traversal of the graph starting from the given root.
     * 
     * @param root The ID of the starting vertex.
     * @return std::vector<int> A vector of vertex IDs in postorder.
     */
    std::vector<int> postorder_traversal(int root) const {
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

    /**
     * @brief Checks if vertex u is an ancestor of vertex v in the given tree.
     * 
     * This is a friend function that determines whether u is an ancestor of v
     * in the directed graph interpreted as a tree.
     * 
     * @param tree The directed graph to check.
     * @param u The potential ancestor vertex ID.
     * @param v The potential descendant vertex ID.
     * @return bool True if u is an ancestor of v, false otherwise.
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

#endif
