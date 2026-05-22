function compute_hdrvdp_pair(mat_path, out_path, v2_root, v3_root, display_inches, viewing_distance_m, peak_luminance)
% Compute official HDR-VDP-2 (Q, 0-100) and HDR-VDP-3 quality (Q_JOD).
% Inputs are linear RGB in cd/m^2 (H x W x 3), saved from Python as test/ref.

if nargin < 7 || isempty(peak_luminance)
    peak_luminance = 1000.0;
end
if nargin < 6 || isempty(viewing_distance_m)
    viewing_distance_m = 0.5;
end
if nargin < 5 || isempty(display_inches)
    display_inches = 24.0;
end

pkg('load', 'image');
pkg('load', 'statistics');

data = load(mat_path);
test = double(data.test);
ref = double(data.ref);
if ndims(test) == 2
    test = repmat(test, [1, 1, 3]);
end
if ndims(ref) == 2
    ref = repmat(ref, [1, 1, 3]);
end
test = max(test, 1e-6);
ref = max(ref, 1e-6);

[h, w, ~] = size(test);
addpath(v3_root);
addpath(fullfile(v3_root, 'utils'));
ppd = hdrvdp_pix_per_deg(display_inches, [w, h], viewing_distance_m);

q2 = NaN;
q3 = NaN;

% --- HDR-VDP-2.2.2 (paper metric: quality correlate Q on linear HDR) ---
try
    addpath(v2_root);
    res2 = hdrvdp(test, ref, 'rgb-bt.709', ppd, {});
    q2 = double(res2.Q);
catch err2
    q2 = NaN;
    warning('HDR-VDP-2 failed: %s', err2.message);
end

% --- HDR-VDP-3.0.7 quality task (Q_JOD) ---
try
    addpath(v3_root);
    addpath(fullfile(v3_root, 'utils'));
    res3 = hdrvdp3('quality', test, ref, 'rgb-native', ppd, {'use_gpu', false});
    if isfield(res3, 'Q_JOD')
        q3 = double(res3.Q_JOD);
    elseif isfield(res3, 'Q')
        q3 = double(res3.Q);
    end
catch err3
    q3 = NaN;
    warning('HDR-VDP-3 failed: %s', err3.message);
end

fid = fopen(out_path, 'w');
if fid < 0
    error('Could not open output file: %s', out_path);
end
fprintf(fid, '{"hdrvdp2":%.8f,"hdrvdp3":%.8f}\n', q2, q3);
fclose(fid);

fprintf(1, 'HDRVDP2=%g\nHDRVDP3=%g\n', q2, q3);
