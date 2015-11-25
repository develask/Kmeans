package distance;

import java.util.ArrayList;

import weka.core.Instance;

public class Minkowski{

	private int p = 3;
	
	public Minkowski(int p) {
		this.p = p>0?p:3;
	}
	public void setP(int p){
		this.p=p;
	}
	
	public double calcularDistancia(Instance bat, Instance bi){
		int numAtrBat = bat.numAttributes();
		int numAtrBi = bi.numAttributes();
		ArrayList<Double> lag = new ArrayList<Double>();
		for(int i=0;i<(numAtrBat<numAtrBi?numAtrBat:numAtrBi);i++){
			lag.add( Math.pow(Math.abs((double)bat.value(i)-(double)bi.value(i)), this.p));
		}
		double sum = 0;
		for (double el: lag) sum += el;
		return Math.pow(sum, 1.0/this.p);
	}

}
