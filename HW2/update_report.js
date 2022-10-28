(async () => {
	const cp = await import('cp-file');

	await cp.copyFile('./document/report.pdf', './report/report.pdf');

	console.log('success update report');
})();
