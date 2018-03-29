export start=6373

for num in $(seq 93 97)
do
  echo $num
  export end=$(($start + 100))
  echo $start, $end
  python changepoint_unkn_3.py ./files/sep.mat $start $end 100 > changepoint_unkn_"$num".out 2> changepoint_unkn_"$num".err &
  start=$(($end + 1))
done

python changepoint_unkn_3.py ./files/sep.mat 6878 7009 10 > changepoint_unkn_98.out 2> changepoint_unkn_98.err &
