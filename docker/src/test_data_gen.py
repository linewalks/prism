from measurement_stat import MEASUREMENT_SOURCE_VALUE_STATS
from data_generator import (generate_measurement, 
                            generate_condition,
                            randomtimes,
                            generate_dev_data)


class Test_DataGenerator():
  def test_generate_measurement(self):
    n = 10
    generated_data = generate_measurement(n)
    assert len(generated_data) == n * len(MEASUREMENT_SOURCE_VALUE_STATS)

  def test_generate_condition(self):
    n = 5
    generated_data = generate_condition(n)
    assert len(generated_data) == n

  def test_generate_time(self):
    n = 10
    generated_data = randomtimes(n=n)
    assert len(generated_data) == n
    
  def test_generate_dev_data(self):
    n = 3
    n_cond = 4
    n_msmt = 3
    generate_dev_data(n, n_cond, n_msmt)
    