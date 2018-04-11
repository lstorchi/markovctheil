export start=4959

for num in $(seq 79 92)
do
  echo $num
  export end=$(($start + 100))
  echo $start, $end
  python changepoint_unkn_3.py ./files/sep.mat $start $end 365 > changepoint_unkn_"$num".out 2> changepoint_unkn_"$num".err &
  start=$(($end + 1))
done

