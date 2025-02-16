# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



function make_CQ_vs_flux_histograms(time_trace_params, itps, i_Ng2; N_bin=200, CQ_range=(-0.3, 1.5))
    bins = range(CQ_range..., N_bin+1)
    N_Bp = length(time_trace_params.sweepPoints[:phi])
    Z = Dict{Int,Any}()
    for parity in [-1,+1]
        traces = generate_time_traces(time_trace_params, itps; fix_parity=parity)
        Z[parity] = zeros(N_bin, N_Bp)
        for i_Bp in 1:N_Bp
            timetrace = real.(traces[:,i_Ng2,i_Bp])
            Z[parity][:,i_Bp] = np.histogram(timetrace, bins=bins)[1]
        end
    end
    return bins, Z
end

function make_CQ_vs_flux_histograms(traces::PyObject, i_Ng2; N_bin=200, CQ_range=(-0.3, 1.5))
    bins = range(CQ_range..., N_bin+1)
    N_Bp = length(traces.phi)
    Z = Dict{Int,Any}()
    for (parity, data) in zip([-1,+1], [traces.CQ_odd, traces.CQ_even])
        Z[parity] = zeros(N_bin, N_Bp)
        for i_Bp in 1:N_Bp
            # -1 to account for python's 0-based indexing
            timetrace = data.isel(ng2=i_Ng2-1, phi=i_Bp-1).squeeze(drop=true).data
            Z[parity][:,i_Bp] = np.histogram(timetrace, bins=bins)[1]
        end
    end
    return bins, Z
end

function mix_histograms(Z)
    red = (1, 0, 0)
    blue = (0, 0, 1)
    white = (1, 1, 1)
    r_r = (Z[+1] ./maximum(Z[+1]))'
    r_b = (Z[-1] ./maximum(Z[-1]))'
    w = 1.0 .- (r_r+r_b)
    w[w.<0.0] .= 0.
    return [w[i].* white .+ r_r[i] .* red .+ r_b[i] .* blue for i in eachindex(r_r)]
end
