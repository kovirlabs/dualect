% Every comment form; the arithmetic shows that none of them was executed.
x = 1;  % trailing comment
# Octave also accepts hash comments
%{
x = 100;
%}
#{
x = 200;
#}
%% a cell divider is a comment too
y = x + 1;  % y = 50 would be ignored here
fmt = '%d items';  % a percent sign inside a string is not a comment
