package src;
import java.util.Hashtable;


public class Args {
	public static Hashtable<String, String> parse(String[] args){
		Hashtable<String, String> r = new Hashtable<String, String>();
		boolean isValue = false;
		String before = "";
		for (String string : args) {
			if (string.indexOf("-")==0){
				r.put(string, "true");
				isValue = true;
			}else{
				if (isValue){
					r.replace(before, string);
				}else{
					r.put(string, "true");
				}
				isValue = false;
			}
			before = string;
		}
		return r;
	}
}
