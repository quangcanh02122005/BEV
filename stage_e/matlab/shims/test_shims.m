%TEST_SHIMS Self-test of the base-MATLAB shims on synthetic data with known ground truth.
rng(1); n = 4000;
xy = (rand(n, 2) - 0.5) * 40; z = 0.02 * randn(n, 1) + 0.10 * xy(:, 1) * 0;    % flat ground z~0, noise 2 cm
ground = [xy z]; out = [(rand(300, 2) - 0.5) * 40, 0.3 + rand(300, 1)];        % 300 obstacle points above ground
model = pcfitplane(pointCloud([ground; out]), 0.1, [0 0 1], 10);
p = model.Parameters; if p(3) < 0, p = -p; end
assert(abs(p(3)) > 0.999 && abs(p(4)) < 0.05, 'plane fit failed: %s', mat2str(p, 4));
fprintf('pcfitplane: normal=%s d=%.4f (truth [0 0 1], 0)\n', mat2str(p(1:3), 4), p(4));
% 3 well separated blobs (40, 25, 12 pts) + 5 isolated points
c1 = [0 0 0] + 0.2 * randn(40, 3); c2 = [10 0 0] + 0.2 * randn(25, 3); c3 = [0 10 0] + 0.2 * randn(12, 3); iso = [20 20 0; -20 20 0; -20 -20 0; 30 0 0; 0 30 0];
L = pcsegdist(pointCloud([c1; c2; c3; iso]), 0.8, 'NumClusterPoints', [6 inf]);
assert(max(L) == 3 && all(L(1:40) == L(1)) && all(L(41:65) == L(41)) && all(L(66:77) == L(66)) && all(L(78:end) == 0), 'clustering failed');
fprintf('pcsegdist: %d clusters (truth 3), isolated points labelled 0\n', max(L));
% chain: 20 points spaced 0.5 m must be ONE cluster at 0.8 m gate (single linkage)
chain = [(0:19)' * 0.5, zeros(20, 2)]; Lc = pcsegdist(pointCloud(chain), 0.8, 'NumClusterPoints', [6 inf]);
assert(all(Lc == 1), 'single-linkage chain failed'); fprintf('pcsegdist: single-linkage chain OK\nSHIM SELFTEST PASS\n');
