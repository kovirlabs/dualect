% Command syntax: every word after the command name is passed as a string.
a = 5;
b = 7;
c = 9;
format long       % display only: no numeric effect
clear a b         % same as clear('a', 'b')
words = 'a b';    % what clear received: text, not values
n_words = numel(strsplit(words, ' '));
format short
