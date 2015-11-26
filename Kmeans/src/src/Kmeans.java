package src;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;

import distance.Minkowski;
import weka.clusterers.Clusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveType;

public class Kmeans implements Clusterer{
	
	private int numCentroides;
	private Instance[] centroides;
	private Instances instancias;
	private Minkowski m;
	private ArrayList<Instance>[] grupos;
	
	private int numIterations;
	private long miliseconds;
	
	public Kmeans(int centroide, int minkowski){
		this.numCentroides = centroide;
		this.m = new Minkowski(minkowski);
		this.numIterations = -1;
		this.miliseconds = -1;
	}
	
	public void setIterations(int num){
		this.numIterations = num;
	}
	public void setTimeOut(long ms){
		this.miliseconds = ms;
	}
	
	private boolean compararCentroides(Instance[] centroides1, Instance[] centroides2){
		if (centroides1 == null || centroides2 == null ) return false;
		Instance instance;
		Instance instance2;
		for (int i = 0; i < centroides1.length; i++) {
			instance = centroides1[i];
			instance2 = centroides2[i];
			if (instance == null || instance2 == null ) return false;
			for (int j = 0; j < instance.numAttributes(); j++) {
				if (instance.value(j) != instance2.value(j)) return false;
			}
		}
		return true;
	}
	@Override
	public void buildClusterer(Instances arg0) throws Exception {
		RemoveType rm = new RemoveType();
		rm.setOptions(new String[]{"-T", "numeric", "-V"});
		rm.setInputFormat(arg0);
		this.instancias = Filter.useFilter(arg0, rm);
		this.centroides = new Instance[this.numCentroides];
		Instance[] centroidesTmp = new Instance[this.numCentroides];
		this.buscarPrimerosCentroides(centroidesTmp);
		grupos = new ArrayList[this.numCentroides];
		int buelta = 0;
		long TInicio = System.currentTimeMillis();
		long TFin = System.currentTimeMillis();
		while(!this.compararCentroides(this.centroides, centroidesTmp) && (this.numIterations!=-1?this.numIterations>buelta++:true) && (this.miliseconds!=-1?TFin-TInicio<this.miliseconds:true)){
			for (int i = 0; i < this.grupos.length; i++) {
				this.grupos[i] = new ArrayList<Instance>();
				
			}
			this.centroides=centroidesTmp;
			for (Instance is : this.instancias) {
				this.grupos[this.clusterInstance(is)].add(is);
			}
			centroidesTmp = this.nuevosCentroides();
//			System.out.println(buelta + " | " + centroidesTmp[0] + " , " + centroidesTmp[1] + " , " +centroidesTmp[2]);
//			System.out.println();
			TFin = System.currentTimeMillis();
		}
	}
	
	private Instance[] nuevosCentroides() {
		Instance[] nuevos = new Instance[this.numCentroides];
		for (int c=0; c<this.grupos.length; c++) {
			ArrayList<Instance> grupo = grupos[c];
			double[] attr = new double[this.instancias.numAttributes()];
			for (Instance ins : grupo) {
				for (int i = 0; i < attr.length; i++) {
					attr[i]+=ins.value(i);
				}
			}
			Instance ins = new DenseInstance(attr.length);
			for (int i = 0; i < attr.length; i++) {
				attr[i]/=grupo.size();
				ins.setValue(i, attr[i]);
			}
			nuevos[c] = ins;
		}
		return nuevos;
	}
	
	private void buscarPrimerosCentroides(Instance[] centroidesTmp){
		int r;
		ArrayList<Integer> nums = new ArrayList<Integer>();
		for (int i = 0; i < this.numCentroides; i++) {
			do{
				r = (int) Math.floor(Math.random() * this.instancias.numInstances());
			}while(nums.contains(r));
			centroidesTmp[i] = this.instancias.get(r);
		}
	}

	@Override
	public int clusterInstance(Instance arg0) throws Exception {
		double[] distancias = new double[this.numCentroides];
		for (int i = 0; i < distancias.length; i++) {
			distancias[i] = this.m.calcularDistancia(arg0, this.centroides[i]);
		}
		double min = Double.MAX_VALUE;
		int pos = -1;
		for (int i=0; i<distancias.length; i++) {
			double d = distancias[i];
			if (min > d){
				min = d;
				pos = i;
			}
		}
		return pos;
	}
	
	public double SSE(){
		double suma=0.0;
		for (int i=0; i<this.grupos.length; i++) {
			ArrayList<Instance> grupo = this.grupos[i];
			for (Instance instance : grupo) {
				suma += Math.pow(this.m.calcularDistancia(instance, this.centroides[i]), 2);
			}
		}
		return suma;
	}
	
	public double silhouette(){
		double sumaA;
		double sumaB;
		double si=0.0;
		for (int i=0; i<this.grupos.length; i++) {
			ArrayList<Instance> grupo = this.grupos[i];
			for (Instance instance : grupo) {
				sumaA = this.calcularDistanciaMediaGrupo(instance, grupo, true);
				sumaB = this.calcularDistanciaMinimaMediaOtrosGrupo(instance, grupo);
				si += (sumaB-sumaA)/(sumaA<sumaB?sumaB:sumaA);
			}
		}
		return si/this.instancias.size();
	}
	private double calcularDistanciaMediaGrupo(Instance ins, ArrayList<Instance> grupo, boolean tuyo){
		double suma=0.0;
		for (Instance instance : grupo) {
			suma+=this.m.calcularDistancia(ins, instance);
		}
		return suma/(grupo.size()-(tuyo?1:0));
	}
	
	private double calcularDistanciaMinimaMediaOtrosGrupo(Instance ins, ArrayList<Instance> grupo){
		double min = Double.MAX_VALUE;
		double tmp;
		for (ArrayList<Instance> grup : this.grupos) {
			if (grupo != grup){
				tmp = this.calcularDistanciaMediaGrupo(ins, grup, false);
				if (tmp<min)min=tmp;
			}
		}
		return min;
	}

	public ArrayList<Instance>[] distributionForCluster() throws Exception {
		return this.grupos;
	}
	
	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int numberOfClusters() throws Exception {
		return this.numCentroides;
	}
	
	public static void main(String[] args) throws Exception {
		Hashtable<String, String> params = Args.parse(args);
		Instances ins = Lector.getLector().leerInstancias(params.get("-f"));
		if (params.get("-class")!=null){
			Remove rm = new Remove();
			rm.setAttributeIndicesArray(new int[]{ins.classIndex()});
			rm.setInputFormat(ins);
			ins = Filter.useFilter(ins, rm);
		}
		String tmp = params.get("-m");
		int t=2;
		switch (tmp!=null?tmp:"3") {
		case "Manhattan":
			t = 1;
			break;
		case "Euclidea":
			t = 2;
			break;
		default:
			if (tmp!=null && Integer.parseInt(tmp)>0){
				t = Integer.parseInt(tmp);
			}
		}
		long TInicio = System.currentTimeMillis();
		Kmeans k = new Kmeans(Integer.parseInt(params.get("-k")),t);
		System.out.println("Se esta generando el cluster");
		k.buildClusterer(ins);
		long TFin = System.currentTimeMillis();
		System.out.println("\nSe ha tardado " + (TFin-TInicio) + " milisegundos en generar los clusters.");
		int i=0;
		System.out.println("\nSe han creado los siguientes clusters:");
		for (ArrayList<Instance> instances : k.distributionForCluster()) {
			System.out.println("\t- Grupo "+ ++i + ":\t" + instances.size() + " instancias.");
		}
		System.out.println("\nSSE: " + k.SSE());
		System.out.println("Silhouette: " + k.silhouette());
	}

}
