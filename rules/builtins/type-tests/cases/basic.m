% Logicals are not numeric; characters are neither; an empty array keeps its class.
x = 3.5;
mask = [1 5 2] > 2;
name = 'beam';
e = [];
num_x = isnumeric(x);
num_mask = isnumeric(mask);     % false
num_name = isnumeric(name);     % false
num_e = isnumeric(e);           % true
log_mask = islogical(mask);
log_x = islogical(x);
log_e = islogical(e);           % false
chr_name = ischar(name);
chr_x = ischar(x);
