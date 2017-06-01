gnuplot --persist << FIN
set key left top
set term jpeg 
plot "accuracy.log" using 3

FIN

