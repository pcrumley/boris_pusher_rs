# boris_pusher_rs
This is part of a larger experimental project of writing a high-performance
pseudo-spectral particle-in-cell code in Rust. This is simply the part of the
code that moves and pushes on the particles following the well known Boris
method, which does a great job at conserving energy as the particle moves in
an arbitrary electric and magnetic field.

This code tracks a single particle as it moves in a constant magnetic field,
saved as an *.npy file in ~/output/trck_prtl. You can plot the trajectory with
python plotter.py script.
