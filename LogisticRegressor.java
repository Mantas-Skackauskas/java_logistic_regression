package LogisticRegressor;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A simple Logistic Regressor by Mantas Skackauskas
 * My python implementation turned into Java code
 */
public class LogisticRegressor {

    private double param;
    private double[] delta;
    private int steps, num_vars;

    /**
     * Creates a Logistic Regressor
     *
     * @param num_vars number of variables
     * @param param    double parameter
     * @param steps    number of steps for gradient descent
     */
    public LogisticRegressor(int num_vars, double param, int steps) {
        this.param = param;
        this.delta = new double[num_vars];
        this.num_vars = num_vars;
        this.steps = steps;
        Arrays.fill(delta, 0.0);
    }

    /**
     * Method used to train the linear regressor
     *
     * @param data   2d double array of rows of double data
     * @param labels 1d double array of labels
     */
    public void train(double[][] data, double[] labels) {
        double[] tmp = new double[num_vars];
        Arrays.fill(tmp, 0.0);
        IntStream.range(0, this.steps).forEachOrdered(i -> {
            IntStream.range(0, this.num_vars)
                    .forEachOrdered(j -> tmp[j] = delta[j] - (param * sum(j, data, labels)));
            IntStream.range(0, this.num_vars)
                    .forEachOrdered(j -> delta[j] = tmp[j]);
            System.out.println("Gradient descent step " + (i + 1));
        });
    }

    /**
     * Sum matrix corresponding elements
     *
     * @param j      index of j-th element
     * @param data   2d double array of rows of double data
     * @param labels 1d double array of labels
     * @return average difference
     */
    private double sum(int j, double[][] data, double[] labels) {
        double sum = IntStream.range(0, data.length)
                .mapToDouble(i -> (predict(data[i]) - labels[i]) * data[i][j]).sum();
        return sum / data.length;
    }
    
    /**
     * Sigmoid
     *
     * @param x_i double number
     * @return double in range (0,1)
     */
    private double sigmoid(double x_i) {
        return 1 / (1 + Math.exp(-x_i));
    }

    /**
     * Method to calculate the probability of given data
     *
     * @param x_i 1d double data element
     * @return probability
     */
    public double predict(double[] x_i) {
        return sigmoid(IntStream.range(0, delta.length)
                .mapToDouble(i -> delta[i] * x_i[i]).sum());
    }
}