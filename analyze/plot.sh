gnuplot --persist << FIN > accuracy-`date "+%m-%d-%y-%H-%M-%S"`.jpeg
set key left top
set term jpeg 
plot "../log/accuracy.log" using 2

FIN

