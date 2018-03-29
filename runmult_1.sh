export start=1

for num in $(seq 1 32)
do
  echo $num
  export end=$(($start + 50))
  echo $start, $end
  python changepoint_unkn_3.py ./files/sep.mat $start $end 1095 > changepoint_unkn_"$num".out 2> changepoint_unkn_"$num".err &
  start=$(($end + 1))
done

