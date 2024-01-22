import numpy as np

from ._ctf import ctf


def _cmaskn_modified(c, ci, a, ai, i):
    H, W = c.shape
    c = c.flatten()
    ci = ci.flatten()

    t = np.where(np.abs(ci) > 1)
    ci[t] = 1
    ai = ai.flatten()
    a = a.flatten()

    ct = ctf(i)
    T = ct * (0.86 * ((c / ct) - 1) + 0.3)

    a1 = np.where((np.abs(ci - c) - T) < 0)
    ai[a1] = a[a1]

    return np.reshape(ai, (H, W))


def _gthresh_modified(x, T, z):
    H, W = x.shape

    x = x.flatten()
    z = z.flatten()

    a = np.where(np.abs(x) < T)
    z[a] = 0

    return np.reshape(z, (H, W))


def nqm(O, I, VA, N):
    x, y = O.shape

    xplane, yplane = np.meshgrid(np.arange(-y / 2, y / 2 - 1, np.arange(-x / 2, x / 2 - 1)))

    plane = xplane + 1j * yplane
    r = np.abs(plane)

    FO = np.fft.fft2(O)
    FI = np.fft.fft2(I)

    # Decompose image with cosine log filter bank
    G_0 = 0.5 * (1 + np.cos(
        np.pi * np.log2((r + 2) * (r + 2 <= 4) * (r + 2 >= 1) + 4 * (~(r + 2 <= 4) * (r + 2 >= 1))) - np.pi))
    G_1 = 0.5 * (1 + np.cos(np.pi * np.log2(r * (r <= 4) * (r >= 1) + 4 * (~(r <= 4) * (r >= 1))) - np.pi))
    G_2 = 0.5 * (1 + np.cos(np.pi * np.log2(r * (r >= 2) * (r <= 8) + 0.5 * (~(r >= 2) * (r <= 8)))))
    G_3 = 0.5 * (1 + np.cos(np.pi * np.log2(r * (r >= 4) * (r <= 16) + 4 * (~(r >= 4) * (r <= 16))) - np.pi))
    G_4 = 0.5 * (1 + np.cos(np.pi * np.log2(r * (r >= 8) * (r <= 32) + 0.5 * (~(r >= 8) * (r <= 32)))))
    G_5 = 0.5 * (1 + np.cos(np.pi * np.log2(r * (r >= 16) * (r <= 64) + 4 * (~(r >= 16) * (r <= 64))) - np.pi))

    GS_0 = np.fft.fftshift(G_0)
    GS_1 = np.fft.fftshift(G_1)
    GS_2 = np.fft.fftshift(G_2)
    GS_3 = np.fft.fftshift(G_3)
    GS_4 = np.fft.fftshift(G_4)
    GS_5 = np.fft.fftshift(G_5)

    L_0 = GS_0 * FO
    LI_0 = GS_0 * FI

    l_0 = np.real(np.fft.ifft2(L_0))
    li_0 = np.real(np.fft.ifft2(LI_0))

    A_1 = GS_1 * FO
    AI_1 = GS_1 * FI

    a_1 = np.real(np.fft.ifft2(A_1))
    ai_1 = np.real(np.fft.ifft2(AI_1))

    A_2 = GS_2 * FO
    AI_2 = GS_2 * FI

    a_2 = np.real(np.fft.ifft2(A_2))
    ai_2 = np.real(np.fft.ifft2(AI_2))

    A_3 = GS_3 * FO
    AI_3 = GS_3 * FI

    a_3 = np.real(np.fft.ifft2(A_3))
    ai_3 = np.real(np.fft.ifft2(AI_3))

    A_4 = GS_4 * FO
    AI_4 = GS_4 * FI

    a_4 = np.real(np.fft.ifft2(A_4))
    ai_4 = np.real(np.fft.ifft2(AI_4))

    A_5 = GS_5 * FO
    AI_5 = GS_5 * FI

    a_5 = np.real(np.fft.ifft2(A_5))
    ai_5 = np.real(np.fft.ifft2(AI_5))

    # Compute contrast images
    c1 = a_1 / l_0
    c2 = a_2 / (l_0 + a_1)
    c3 = a_3 / (l_0 + a_1 + a_2)
    c4 = a_4 / (l_0 + a_1 + a_2 + a_3)
    c5 = a_5 / (l_0 + a_1 + a_2 + a_3 + a_4)

    ci1 = ai_1 / li_0
    ci2 = ai_2 / (li_0 + ai_1)
    ci3 = ai_3 / (li_0 + ai_1 + ai_2)
    ci4 = ai_4 / (li_0 + ai_1 + ai_2 + ai_3)
    ci5 = ai_5 / (li_0 + ai_1 + ai_2 + ai_3 + ai_4)

    # Detection Thresholds
    d1 = ctf(2 / VA)
    d2 = ctf(4 / VA)
    d3 = ctf(8 / VA)
    d4 = ctf(16 / VA)
    d5 = ctf(32 / VA)

    # Account for suprathrshold effects (See Bradley and Ozhawa)
    ai_1 = _cmaskn_modified(c1, ci1, a_1, ai_1, 1)
    ai_2 = _cmaskn_modified(c2, ci2, a_2, ai_2, 2)
    ai_3 = _cmaskn_modified(c3, ci3, a_3, ai_3, 3)
    ai_4 = _cmaskn_modified(c4, ci4, a_4, ai_4, 4)
    ai_5 = _cmaskn_modified(c5, ci5, a_5, ai_5, 5)

    # Apply detection thresholds
    A_1 = _gthresh_modified(c1, d1, a_1)
    AI_1 = _gthresh_modified(ci1, d1, ai_1)
    A_2 = _gthresh_modified(c2, d2, a_2)
    AI_2 = _gthresh_modified(ci2, d2, ai_2)
    A_3 = _gthresh_modified(c3, d3, a_3)
    AI_3 = _gthresh_modified(ci3, d3, ai_3)
    A_4 = _gthresh_modified(c4, d4, a_4)
    AI_4 = _gthresh_modified(ci4, d4, ai_4)
    A_5 = _gthresh_modified(c5, d5, a_5)
    AI_5 = _gthresh_modified(ci5, d5, ai_5)

    # reconstruct images
    y1 = A_1 + A_2 + A_3 + A_4 + A_5
    y2 = AI_1 + AI_2 + AI_3 + AI_4 + AI_5

    # compute SNR
    square_err = (y1 - y2) * (y1 - y2)
    np_sum = np.sum(np.sum(square_err))

    sp_sum = np.sum(np.sum(y1 ** 2))

    return 10 * np.log10(sp_sum / np_sum)
