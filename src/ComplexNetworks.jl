using SparseArrays
using Erdos

function barabasi(n::Int, mean_degree::T) where T<:Real
    g = barabasi_albert(n,10);
    return adjacency_matrix(g);
end;

function erdos(n:Int, mean_degree::T) where T<:Real
    p = mean_degree/n;
    g = erdos_reyni(n,p)
    return adjacency_matrix(g);
end;

function random_digraph(n::Int, mean_degree::T) where T<:Real
    p = mean_degree/n;
    g = sprand()
end;