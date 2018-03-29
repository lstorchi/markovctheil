export start=3646

for num in $(seq 66 78)
do
  echo $num
  export end=$(($start + 100))
  echo $start, $end
  python changepoint_unkn_3.py ./files/sep.mat $start $end 1000 > changepoint_unkn_"$num".out 2> changepoint_unkn_"$num".err &
  start=$(($end + 1))
done

