using Plots, MAT

responses = matread("Training_Data.mat")["responses"]

# Data Extraction and Conversion
# Assuming 'responses' is an array of arrays, similar to cell array in MATLAB
S11_val = responses[36]  # response of interest (specify it accordingly)

# Select the complex values of S11 in S11_val
complex_S11_temp = S11_val[:, 2:3]

# Convert the selected values from double to complex
complex_S11 = complex_S11_temp[:, 1] + complex_S11_temp[:, 2] * im

# Derive the magnitude or absolute value of the complex values
S11_abs = abs.(complex_S11)

# Convert to decibels
S11_dB = 20 * log10.(S11_abs)

# Plot

# Note the bounds and step size in the frequency points
ω = 4.5e9:0.00199990000000039e9:6.5e9

# Frequency unit in GHz
ωf = ω ./ 1e9
p = plot(ωf, S11_dB, color=:red, linewidth=2.5, label="S11 (dB)")
xlabel!("Freq. in GHz")
ylabel!("S_1_1 in dB")
display(p)
readline()
