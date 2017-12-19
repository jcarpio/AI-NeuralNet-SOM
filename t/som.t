######################### We start with some black magic to print on failure.

BEGIN { $| = 1; print "1..8\n"; }
END {print "not ok 1\n" unless $loaded;}
use AI::NeuralNet::SOM;
use locale;
$loaded = 1;

print "ok 1\n";
######################### End of black magic.

#########################
#
# Create the SOM object (map)
#
#########################
$som = AI::NeuralNet::SOM->new(); 
$out = ($som)?"ok 2":"not ok 2";
print "$out\n";

#########################
#
# Loading data and initialize map
#
#########################
@data = ();
open(DATA, "t/ex.dat");
<DATA>;
while (<DATA>) {
	chomp();
	@pattern = split(/ /);
	push(@data, @pattern);
}
close (DATA);

$som->initialize(5,5,5,'hexa','bubble','linear',0,\@data);
$qerr1 = $som->qerror(\@data);
print "ok 3\n";

#########################
#
# Train the map
#
#########################
$som->train(2000,0.1,5,'linear',\@data);
$qerr2 = $som->qerror(\@data);
$out = "not ok 4";
$qerr3 = $som->qerror(\@data);
if ($qerr2 < $qerr1) {
	$out = "ok 4";
}
print "$out\n";

#########################
#
# Loading data and label map
#
#########################
open(DATA, "t/ex_fts.dat");
<DATA>;
while(<DATA>) {
	chomp();
	@pattern = split(/ /);
	($x, $y) = $som->winner(\@pattern);
	$som->set_label($x, $y, $pattern[$som->i_dim()]);
}
close (DATA);
print "ok 5\n";

#########################
#
# Test map with some fault patterns
#
#########################
$out = "not ok 6";
open(DATA, "t/ex_fdy.dat") or die;
<DATA>;
while(<DATA>) {
	chomp();
	@pattern = split(/ /);
	($x, $y) = $som->winner(\@pattern);
	$label=$som->label($x, $y);
	if (defined($label) and $label eq 'fault') {
		$out = "ok 6";
	}
}
close (DATA);
print "$out\n";

#########################
#
# Test map with all normal patterns
#
#########################
$out = "ok 7";
open(DATA, "t/ex_ndy.dat") or die;
<DATA>;
while(<DATA>) {
	chomp();
	@pattern = split(/ /);
	($x, $y) = $som->winner(\@pattern);
	$label=$som->label($x, $y);
	if (defined($label) and $label eq 'fault') {
		$out = "not ok 7";
	}
}
close (DATA);
print "$out\n";

#########################
#
# Save and load map
#
#########################
open (CODE, ">t/test.cod") or die;
$som->save(*CODE);
close (CODE);

open (CODE, "t/test.cod") or die;
$som->load(*CODE);
close (CODE);

$out = "not ok 8";
$qerr3 = $som->qerror(\@data);
if ($qerr3 = $qerr2) {
	$out = "ok 8";
}
print "$out\n";

unlink("t/test.cod");
